import torch
from torchvision import transforms
from PIL import Image

def model_predict(image_path, model, device, topk):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    Parameters:
     image_path: String specifying the file path of the image
     model: PyTorch model to use
     device: String specifying which device (CPU or GPU) to use
     topk: Integer specifying number of top K classes to output
    Returns:
     top_class: List containing the likeliest categories of the image
    '''
    infer_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    model.to(device)
    model = model.eval()
    # print(next(model.parameters()).is_cuda) # check whether the model is on GPU if GPU is used
    with Image.open(image_path) as im:
        image = infer_transforms(im).unsqueeze(0)  
        image = image.to(device)
        # print(image.is_cuda) # check whether the image is on GPU if GPU is used
        output = model.forward(image)
        ps = torch.exp(output)
        top_class = ps.topk(topk, dim=1)[1]
        return top_class.tolist()[0]