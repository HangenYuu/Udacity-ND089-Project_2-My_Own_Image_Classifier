import argparse

def get_input_args():
    '''
    Retrieves and parses the 7 command line arguments (1 compulsory and 6 optional
    provided by the user when they run train.py from a terminal window. This function
    uses Python's argparse module. If the user fails to provide some or all of the 6 
    optional arguments, the default values are used for the missing arguments. 
    Command Line Arguments:
      1. Data Directory as data_dir.
      2. Directory to Save Checkpoints as --save_dir with default value 'save_directory'.
      3. Model Architecture as --arch with default value 'vgg13'.
      4. (Hyperparameters) Learning Rate as --learning_rate with default value 0.01.
      5. (Hyperparameters) Number of Hidden Units as --hidden_units with default value 512.
      6. (Hyperparameters) Number of Training Epochs as --epochs with default value 20.
      7. Use GPU for Training with argument --gpu.
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments object
     '''
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--arch', default='vgg13')
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--hidden_units', default=512)
    parser.add_argument('--epochs', default=20)
    parser.add_argument('--gpu')
    
    return parser.parse_args()