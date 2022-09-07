import torch
from torch import optim, nn

def train_model(model, arch, trainloader, validloader, testloader, learning_rate, epochs, device, save_dir, input_to_classifier, hidden_units):
    """
    Train the neural network model on the datasets with the specified hyperparameters, and then save the model
    to the specified folder.
    Parameters:
     model - PyTorch model to be used.
     arch - String specifying the architecture of the model
     trainloader, validloader, testloader - Image DataLoaders to train the model.
     learning_rate - Integer specifying the learning rate
     epochs - Integer specifying the number of epochs to train the model
     device - String specifying the device (CPU or GPU) to train the model
     save_dir - String specifying where to save the model
     input_to_classifier - Number of input to the new feedforward classifier
     hidden_units - Number of hidden_units of the new feedforward classifier
    Returns:
     None - Simply printing training loss, validation loss, validation accuracy every 5 batches, and the final test accuracy
    """
    if arch == 'resnet50':
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    model.to(device)
    step = 0 #In one epoch there is multiple steps
    running_loss = 0
    print_every = 5

    for e in range(epochs):
        for images, labels in trainloader:
            step += 1
        
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if step % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
            
                for images, labels in validloader:
                
                    images, labels = images.to(device), labels.to(device)
                
                    logps = model(images)
                    loss = criterion(logps, labels)
                    valid_loss += loss.item()
                
                    ps = torch.exp(logps)
                    top_class = ps.topk(1, dim=1)[1]
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {e+1}/{epochs}).. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {valid_loss/len(validloader):.3f}.. "
                      f"Test accuracy: {accuracy*100/len(validloader):.1f}%")
            
                running_loss = 0
                model.train()
    
    model.eval()
    test_loss = 0
    accuracy = 0
            
    for images, labels in testloader:
                
        images, labels = images.to(device), labels.to(device)
                
        logps = model(images)
        loss = criterion(logps, labels)
        test_loss += loss.item()
                
        ps = torch.exp(logps)
        top_class = ps.topk(1, dim=1)[1]
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy*100/len(testloader):.1f}%")

    checkpoint = {'architecture': arch,
                  'input_to_classifier': input_to_classifier,
                  'hidden_units': hidden_units,
                  'model_state_dict': model.state_dict(),
                  'epochs': epochs,
                  'optimizer_state_dict': optimizer.state_dict()}
    filepath = save_dir + '/checkpoint.pth' # This will overwrite on the checkpoint.pth file here.
    torch.save(checkpoint, filepath)