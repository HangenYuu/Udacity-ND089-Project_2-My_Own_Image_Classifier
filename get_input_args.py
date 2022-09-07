import argparse

def get_input_train():
    '''
    Retrieves and parses the 7 command line arguments (1 compulsory and 6 optional)
    provided by the user when they run train.py from a terminal window. This function
    uses Python's argparse module. If the user fails to provide some or all of the 6 
    optional arguments, the default values are used for the missing arguments. 
    Command Line Arguments:
      1. Data Directory as data_dir.
      2. Directory to Save Checkpoints as --save_dir with default value 'save_directory'.
      3. Model Architecture as --arch, 3 values 'vgg13', 'resnet50' or 'densenet121' with default value 'densenet121'.
      4. (Hyperparameters) Learning Rate as --learning_rate with default value 0.01.
      5. (Hyperparameters) Number of Hidden Units as --hidden_units with default value 256.
      6. (Hyperparameters) Number of Training Epochs as --epochs with default value 3.
      7. Use GPU for Training if Available as --gpu with default value False.
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None
    Returns:
     parse_args() - data structure that stores the command line arguments object
     '''
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Specifying directory of the training data')
    parser.add_argument('--save_dir', default='checkpoints', help='Specifying where to save the models')
    parser.add_argument('--arch', default='densenet121', help='Choose among the 3 model architectures - DenseNet, VGG, or ResNet')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Specifying learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=512, help='Specifying the number of hidden units in the customized classifier network')
    parser.add_argument('--epochs', type=int, default=3, help='Specifying the number of training epochs')
    parser.add_argument('--gpu', action='store_true', default=False, help='Specifying whether to use GPU for training or not (recommend)')
    
    return parser.parse_args()

def get_input_predict():
    '''
    Retrieves and parses the 5 command line arguments (2 compulsory and 3 optional)
    provided by the user when they run train.py from a terminal window. This function
    uses Python's argparse module. If the user fails to provide some or all of the 3 
    optional arguments, the default values are used for the missing arguments. 
    Command Line Arguments:
      1. Path to Image as image_path.
      2. Path to Pre-trained Model for Inference as checkpoint.
      3. Return Top K Most Likely Classes as --top_k with default value 5.
      4. Load File to Map Categories to Real Names as --category_names with default value 'cat_to_name.json'.
      5. Use GPU for Inference if Available as --gpu with default value False.
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None
    Returns:
     parse_args() - data structure that stores the command line arguments object
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Specifying image path for inference in the form `/path/to/image`')
    parser.add_argument('checkpoint', type=str, help='Specifying the file path to pre-trained model for inference')
    parser.add_argument('--top_k', default=5, type=int, help='Specifying the number of most likely classes to return')
    parser.add_argument('--category_names', default='cat_to_name.json', type=str, help='Specifying the file used to map categories to real names')
    parser.add_argument('--gpu', action='store_true', default=False, help='Specifying whether to use GPU for training or not (recommend)')
    
    return parser.parse_args()