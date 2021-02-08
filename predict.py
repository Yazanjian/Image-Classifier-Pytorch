from dependencies import *
from utility import * 
from model import load_checkpoint


def predict(image_path, model, topk, category_names, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = process_image(image_path, mean, std)
    img = torch.from_numpy(img).type(torch.FloatTensor)
  
    #Make a batch with a single image
    img = img.unsqueeze_(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else 'cpu')
    model = model.to(device)
    img = img.to(device)

    model.eval()
    with torch.no_grad():
        #Predict 
        logps = model.forward(img)
        ps = torch.exp(logps)

        # get the top k predictions and transfer the results into lists 
        top_p, top_class = ps.topk(topk,  dim=1)
        top_class = top_class.numpy().tolist()[0]
        top_p = top_p.numpy().tolist()[0]

        #Get the idx to class dictionary from the class_to_idx dictionary 
        class_to_idx = model.class_to_idx
        classes = []
        idx_to_class= {x: y for y, x in class_to_idx.items()}
        for item in top_class:
            classes.append(idx_to_class[item])
            
        if category_names != None:
            cat_to_name = parse_json_categories(category_names)
            flower_class_list =  []
            for flower_class in classes: 
                flower_class_list.append(cat_to_name[flower_class])
            return top_p, flower_class_list
        else: 
            None
            
    return top_p, classes


def main():
    #Parsing command line args
    parser = argparse.ArgumentParser(description='Predict the name and prob of a flower.')
    parser.add_argument('image_path', action="store", help='The path of image: e.g. ./flowers/test/58/image_02663.jpg')
    parser.add_argument('checkpoint', action="store", help='The model checkpoint')
    parser.add_argument('--topk',  dest="topk", action="store", type=int, help='top k predictions, default=5', default=5)
    parser.add_argument('--category_names', dest="category_names", action="store", default=None)
    parser.add_argument('--gpu',  dest="gpu", action="store_true", help='Using GPU for model training', default=False)
    
    
    results = parser.parse_args()
    
    model, _, _ = load_checkpoint(results.checkpoint)
    top_p, classes = predict(results.image_path, model, results.topk, results.category_names, results.gpu)
    
    print('The top probs {}, and their clsses {}'.format(top_p, classes))
    
if __name__ == "__main__":
    main()
