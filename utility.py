#Functions to load and print datasets (images)
from dependencies import json, Image, transforms
    
    

def parse_json_categories(file_name):
    with open(file_name, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name
    
     
    
def process_image(image_path, mean, std):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # SOLVING BY TORCHVISION.TRANSFORMS and PIL 
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,std)])
    
    image = transform(image)
    np_image = image.numpy()
    return np_image