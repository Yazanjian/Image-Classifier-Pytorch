
from dependencies import *
from utility import parse_json_categories
from model import get_data, model_traning


def main():
    #Parsing command line args
    parser = argparse.ArgumentParser(description='Train a deep nueral network on a dataset.')

    parser.add_argument('data_directory', action="store", help='Dataset directory: e.g. ./flower/')
    parser.add_argument('--save_dir',  dest="save_dir", action="store", help='Directory to save the checkpoint: e.g. ./Checkpoints',             default='./home/workspace/ImageClassifier/Checkpoints')
    parser.add_argument('--arch',  dest="arch", action="store", help='Arch for transfer learning, supported options: ["vgg16", "vgg16_bn", "vgg13"', default='"vgg16_bn"')
    parser.add_argument('--learning_rate',  dest="learning_rate", action="store", type=float, help='The model learning rate, default=0.01', default=0.001)
    parser.add_argument('--hidden_units',  dest="hidden_units", action="store", type=int, help='The size of the hidden layers, default=512', default=512)
    parser.add_argument('--epochs',  dest="epochs", action="store", type=int, help='Number of epochs, default=5', default=5)
    parser.add_argument('--gpu',  dest="gpu", action="store_true", help='Using GPU for model training', default=False)

    results = parser.parse_args()
    

   
    #Build the model
    arch = results.arch.replace('"', '')
    number_of_output_classes = 102
    #Mean and std for image normalization 
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    model_traning(results.data_directory, means, stds, arch, results.gpu, number_of_output_classes, results.hidden_units, results.learning_rate, results.epochs)
    
    
    
    
if __name__ == "__main__":
    main()
