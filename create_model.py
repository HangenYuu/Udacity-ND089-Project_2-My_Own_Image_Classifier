from torchvision import models
from torch import nn


def create_model(arch, hidden_units):
    """
    Create a model based on the architecture specified by the user.
    Parameters:
     arch - String having the value of the architecture of the model.
     hidden_units - String having the value of the hidden units of the classifier
     feedforward network
    Returns:
     model - PyTorch model network used for training
    """
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_to_classifier = 25088
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        input_to_classifier = 2048
    else:
        model = models.densenet121(pretrained=True)
        input_to_classifier = 1024

    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(input_to_classifier, hidden_units),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(hidden_units, 102),
                           nn.LogSoftmax(dim=1))
    if arch == 'resnet50':
        model.fc = classifier
    else:
        model.classifier = classifier
    return model