#Several functions to build and train deep neural network model
from dependencies import *

def get_data(data_directory, means, stds, batch_sz=64):
    '''
    Define transforms, load the dataset, and return a dataloader with batch size 64
    Inputs: 
    train_dir: a dir path for training set
    means: a mean list for normalization 
    stds: a std list for normalization 
    batch_sz: a batch size,
    '''
    data_dir = data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    data_transforms = {'train_transform': transforms.Compose([transforms.RandomRotation(30),
                                                              transforms.RandomResizedCrop(224),
                                                              transforms.RandomHorizontalFlip(),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(means,stds)]),
                       'valid_transform': transforms.Compose([ transforms.Resize(255),
                                                         transforms.CenterCrop(224),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(means,stds)])}

    # TODO: Load the datasets with ImageFolder
    image_datasets = {'train_data': datasets.ImageFolder(train_dir, transform=data_transforms['train_transform']), 
                      'valid_data': datasets.ImageFolder(valid_dir, transform=data_transforms['valid_transform'])}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'train_loader': torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=batch_sz, shuffle=True),
                   'valid_loader': torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=batch_sz)}
    
    return image_datasets, dataloaders


#Define a classifier 
class Classifier(nn.Module):
    #building a classifier with dynamic number of hidden layers 
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2):
        super().__init__()
        #Create the input layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        #Create hidden layers
        layers_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layers_sizes])
        
        #Create the output layer
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        #Using dropout to reduce overfitting
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        
        return x

# TODO: Save the checkpoint 
def save_checkpoint(model, optimizer, input_size, output_size, train_data, epochs, file_path='checkpoint.pth'):
    '''
    This function save a checkpoint for a model
    
    Param:
    model: the trained model, to extract the hidden layers and state_dict of the model
    optimizer: to save the state of the optimizer 
    input_size: the size of the fully connected nn input
    output_size: the size of the fully connected nn output
    file_path: the path of the file that will have the saved checkpoint
    train_data: the training imgs to save the mapping of classes to indices
    
    Output: N/A
    '''
    print("Inside saving checkpoint")
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
                  'state_dict':model.state_dict(),
                  'class_to_idx': train_data.class_to_idx,
                  'epochs': epochs,
                  'optimizer_state_dict':optimizer.state_dict}
    torch.save(checkpoint, file_path)

    
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(file_path, arch='vgg16_bn'):
    '''
    Load the checkpoint, download the model using the arch, then add the trained classifier to the downloaded model
    '''
    
    checkpoint = torch.load(file_path, map_location="cpu")
    model, _ = download_model(arch)
   
    model.classifier = Classifier(checkpoint['input_size'], checkpoint['output_size'], checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer_state_dict']
    epochs = checkpoint['epochs']
    return model, optimizer, epochs    
    
    
    
#Train the model
def train_validate(model, trainloader, validloader, criterion, optimizer, epochs, print_every=50, using_gpu=True):
    device = torch.device("cuda" if torch.cuda.is_available() and using_gpu else "cpu")
    with active_session():
        steps = 0
        for epoch in range(epochs):
            running_loss = 0
            for inputs, labels in trainloader:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0 
                    accuracy = 0 
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)

                            logps = model.forward(inputs)
                            loss = criterion(logps, labels)

                            valid_loss += loss.item()

                            #Calculate Accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1,  dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                        print(f"Epoch {epoch+1}/{epochs}.. "
                              f"Train loss: {running_loss/print_every:.3f}.. "
                              f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                              f"Validation accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0 
                    model.train()
    return model, optimizer

def download_model(arch='vgg16_bn'):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)     
    elif arch == 'vgg16_bn':        
        model = models.vgg16_bn(pretrained=True)         
    elif arch == 'vgg13':        
        model = models.vgg13(pretrained=True)                 

    input_size = model.classifier[0].in_features
    #Freeze prameters -- keep the features' weights not touched 
    for param in model.parameters():
        param.requires_grad = False
    
    return model, input_size



    
def model_traning(data_directory, means, stds, arch, using_gpu, number_of_output_classes, hidden_units, learning_rate, epochs):
    model, input_size = download_model(arch)
    image_datasets, dataloaders = get_data(data_directory, means, stds)
    #To use GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() and using_gpu else "cpu")
    
    print("The device used is {}".format(device))
    
    #To use the classifier with the defind arch above
    model.classifier = Classifier(input_size, number_of_output_classes, [hidden_units]) 
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    epochs = epochs
    #Move the model data to GPU or CPU device
    model.to(device)
    
    
    print("Training the model...")
    model, optimizer =  train_validate(model, dataloaders['train_loader'], dataloaders['valid_loader'], criterion, optimizer, epochs, 100, using_gpu)
    print(model)
    
    #Save a checkpoint for the classifier only
    print("saving the checkpoint...")
    save_checkpoint(model, optimizer, input_size, number_of_output_classes, image_datasets['train_data'], epochs, 'checkpoint_terminal.pth')   
    
    print("Loading the checkpoint...")
    load_checkpoint('checkpoint_terminal.pth', arch)
    
