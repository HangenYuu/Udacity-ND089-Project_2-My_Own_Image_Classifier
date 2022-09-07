import torch

from get_input_args import get_input_train
from load_dataset import load_dataset
from create_model import create_model
from train_model import train_model
# from workspace_utils import active_session (on Udacity only)

def main():
    # Return the arguments that the user provides
    in_arg = get_input_train()

    # Load the training dataset:
    train_set, valid_set, test_set = load_dataset(in_arg.data_dir)

    # Initialize the model with the new classifier
    input_to_classifier, model = create_model(in_arg.arch, in_arg.hidden_units)

    # Set device for training
    device = torch.device('cuda' if in_arg.gpu and torch.cuda.is_available() else 'cpu')
    
    # Train the model, printing training loss, validation loss, validation accuracy every 5 batches, then save the model to the folder
    # with active_session():
    #     train_model(model, in_arg.arch, train_set, valid_set, test_set, in_arg.learning_rate, in_arg.epochs, device, in_arg.save_dir, input_to_classifier, in_arg.hidden_units)
    train_model(model, in_arg.arch, train_set, valid_set, test_set, in_arg.learning_rate, in_arg.epochs, device, in_arg.save_dir, input_to_classifier, in_arg.hidden_units)    

if __name__ == "__main__":
    main()