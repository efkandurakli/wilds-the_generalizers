import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from utils import move_to
from wilds.common.utils import split_into_groups
from utils import concat_input
from losses import initialize_loss

def reconstruction_loss(original_image, reconstructed_image):
    mse_loss = nn.MSELoss()
    loss = mse_loss(reconstructed_image, original_image)
    return loss

def coral_penalty(x, y):
    if x.dim() > 2:
        # featurizers output Tensors of size (batch_size, ..., feature dimensionality).
        # we flatten to Tensors of size (*, feature dimensionality)
        x = x.view(-1, x.size(-1))
        y = y.view(-1, y.size(-1))

    mean_x = x.mean(0, keepdim=True)
    mean_y = y.mean(0, keepdim=True)
    cent_x = x - mean_x
    cent_y = y - mean_y
    cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
    cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

    mean_diff = (mean_x - mean_y).pow(2).mean()
    cova_diff = (cova_x - cova_y).pow(2).mean()

    return mean_diff + cova_diff

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18()
        self.model.fc = nn.Identity()
        
    def forward(self, x):
        return self.model(x)
    

class DomainClassifier(nn.Sequential):

    def __init__(
        self, in_feature: int, n_domains, hidden_size: int = 512
    ):

        super(DomainClassifier, self).__init__(
            nn.Linear(in_feature, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_domains),
        )


class Decoder(nn.Module):
    def __init__(self, content_dim=512, style_dim=512, output_channels=3, image_size=96):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(content_dim + style_dim, 512 * (image_size // 16) * (image_size // 16))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid() 
        )
        
        self.image_size = image_size

    def forward(self, content_features, style_features):
        combined_features = torch.cat([content_features, style_features], dim=1) 

        x = self.fc(combined_features)  
        x = x.view(-1, 512, self.image_size // 16, self.image_size // 16) 
        
        reconstructed = self.decoder(x) 
        
        return reconstructed

        
        
class DGMedIA(SingleModelAlgorithm):
    
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps, n_domains, group_ids_to_domains):
        model = initialize_model(config, d_out)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.group_ids_to_domains = group_ids_to_domains.to(self.device)
        
        self.content_encoder = Encoder().to(self.device)
        self.style_encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.domain_classifier = DomainClassifier(in_feature=512, n_domains=n_domains).to(self.device)
        
        self.awl = AutomaticWeightedLoss(4).to(self.device)
        
    
    def process_batch(self, batch, unlabeled_batch=None):
        x, y_true, metadata = batch
        
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)
                
        domains_true = self.group_ids_to_domains[g]
        self.domain_loss = initialize_loss('cross_entropy', None)

        content = F.instance_norm(x)
        style = x - content
        
        
        outputs = self.get_model_output(x, y_true)
        
        content_features = self.content_encoder(content)
        style_features = self.style_encoder(style)
        
        reconstructed = self.decoder(content_features, style_features)
        
        domains_pred = self.domain_classifier(style_features)
        
        results = {
            'input': x,
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
            'content_features': content_features,
            'style_features': style_features,
            'reconstructed_image': reconstructed,
            'domains_true': domains_true,
            'domains_pred': domains_pred
            
        }
        return results
        
    def objective(self, results):
        if self.is_training:
            input = results.pop('input')
            features = results.pop('content_features')
            reconstructed_image = results.pop('reconstructed_image')

            # Split into groups
            groups = concat_input(results['g'], results['unlabeled_g']) if 'unlabeled_g' in results else results['g']
            unique_groups, group_indices, _ = split_into_groups(groups)
            n_groups_per_batch = unique_groups.numel()

            # Compute penalty - perform pairwise comparisons between features of all the groups
            penalty = torch.zeros(1, device=self.device)
            for i_group in range(n_groups_per_batch):
                for j_group in range(i_group+1, n_groups_per_batch):
                    penalty += coral_penalty(features[group_indices[i_group]], features[group_indices[j_group]])
            if n_groups_per_batch > 1:
                penalty /= (n_groups_per_batch * (n_groups_per_batch-1) / 2) # get the mean penalty

            rec_loss = reconstruction_loss(input, reconstructed_image)
                        
            domain_classification_loss = self.domain_loss.compute(
                results.pop("domains_pred"),
                results.pop("domains_true"),
                return_dict=False,
            )
            
            classification_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)

            self.save_metric_for_logging(
                results, "classification_loss", classification_loss
            )
            self.save_metric_for_logging(
                results, "domain_classification_loss", domain_classification_loss
            )
            
            self.save_metric_for_logging(
                results, "reconstruction_loss", rec_loss
            )
            
            self.save_metric_for_logging(
                results, "coral_loss", penalty
            )
            
            avg_loss = self.awl(classification_loss, domain_classification_loss, rec_loss, penalty[0])
            
            return avg_loss
        
        classification_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
        return classification_loss
