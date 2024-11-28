import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from torch.nn import functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    
    """

    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    
    """
    
    """
    
    def __init__(self, lambda_):    
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class FeatureExtractor(nn.Module):
    
    """
    
    """
    
    def __init__(self, backbone):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone
        self.embedding_dim = backbone.config.hidden_size

    def forward(self, x):
        outputs = self.backbone(x)
        embeddings = outputs.logits
        
        return embeddings
    
    
class ClinicalOutcomePredictor(nn.Module):
    
    """
    
    """
    
    def __init__(self, embedding_dim, num_outcomes):        
        super(ClinicalOutcomePredictor, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_outcomes)

    def forward(self, x):
        return self.fc(x)


class Adversary(nn.Module):
    
    """
    
    """
    
    def __init__(self, embedding_dim, num_protected_attributes):
        super(Adversary, self).__init__()
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.fc = nn.Linear(embedding_dim, num_protected_attributes)

    def forward(self, x):
        x = self.grl(x)
        return self.fc(x)


def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch]) 
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long) 
    return images, labels