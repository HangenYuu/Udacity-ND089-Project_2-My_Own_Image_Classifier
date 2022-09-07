import json
import torch
from get_input_args import get_input_predict
from load_model import load_checkpoint
from model_predict import model_predict
def main():
    # Return the arguments that the user provides
    in_arg = get_input_predict()
    
    # Load model
    model = load_checkpoint(in_arg.checkpoint)
    
    # Loading necessary information to translate between index, category, and name
    with open('idx_to_class.json', 'r') as fp:
        idx_to_class = json.load(fp)
        
    with open(in_arg.category_names, 'r') as fp:
        cat_to_name = json.load(fp)
    
    # Load device
    device = torch.device('cuda' if in_arg.gpu and torch.cuda.is_available() else 'cpu')
    print(device)
    # Perform inference on the image
    most_likely_class = model_predict(in_arg.image_path, model, device, in_arg.top_k)
    class_list = [idx_to_class[str(i)] for i in most_likely_class]
    most_likely_name = [cat_to_name[i] for i in class_list]
    
    # Print out results
    print(most_likely_class)
    print(most_likely_name)
    
if __name__ == "__main__":
    main()