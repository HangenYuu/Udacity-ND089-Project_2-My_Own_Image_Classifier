import torch
from torchvision import models
from torch import nn

def load_checkpoint(filepath):
    '''
    Load pre-trained model from the file path provided.
    Parameters:
     filepath - String specifying the file path to load model from
    Returns:
     model - PyTorch model
    '''
    checkpoint = torch.load(filepath, map_location='cpu')
    
    if checkpoint['architecture'] == 'vgg13':
        model = models.vgg13(pretrained=False)
    elif checkpoint['architecture'] == 'resnet50':
        model = models.resnet50(pretrained=False)
    else:
        model = models.densenet121(pretrained=False)
    
    classifier = nn.Sequential(nn.Linear(checkpoint['input_to_classifier'], checkpoint['hidden_units']),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(checkpoint['hidden_units'], 102),
                           nn.LogSoftmax(dim=1))
    if checkpoint['architecture'] == 'resnet50':
        model.fc = classifier
    else:
        model.classifier = classifier
        
    model.load_state_dict(checkpoint['model_state_dict'])        
    return model