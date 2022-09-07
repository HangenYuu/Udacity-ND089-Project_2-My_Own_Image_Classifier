import torch
from torchvision import datasets, transforms

def load_dataset(data_dir):
    '''
    Load the training and validation datasets from the data directory. The data directory needs
    to be named in the format required by PyTorch.
    This function returns the two datasets in the form of DataLoader for training the model.
    Parameters:
     data_dir - String having the value of the directory path of the dataset
    Returns:
     train_dataset, valid_dataset, test_set - Image DataLoaders that are used to train the model.
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomRotation(35),
                                      transforms.RandomVerticalFlip(0.17),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_n_test_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_n_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform = valid_n_test_transforms)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    
    return trainloader, validloader, testloader