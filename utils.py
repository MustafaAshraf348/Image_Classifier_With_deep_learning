import torch
from torchvision import datasets, transforms
from PIL import Image
import os

def preprocessing_images(data_dir):
    data_transforms = {
            'train': transforms.Compose([
                     transforms.RandomRotation(30),
                     transforms.RandomResizedCrop(224), 
                     transforms.RandomHorizontalFlip(), 
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], 
                                         [0.229, 0.224, 0.225])
                    ]),
            'valid': transforms.Compose([
                     transforms.Resize(255),
                     transforms.CenterCrop(224),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], 
                                         [0.229, 0.224, 0.225])
                    ]),
            'test': transforms.Compose([
                    transforms.Resize(255),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
                    ])
    }

    data_types = ['train', 'valid', 'test']
    
    image_datasets = {data: datasets.ImageFolder(os.path.join(data_dir, data), 
                                                 transform=data_transforms[data]) 
                      for data in data_types}
    
    dataloaders = {data: torch.utils.data.DataLoader(image_datasets[data], 
                                                     batch_size=64, 
                                                     shuffle=True) 
                   for data in data_types}
    
    dataset_sizes = {data: len(image_datasets[data]) for data in data_types}
    
    return dataloaders, dataset_sizes

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
  
    pil_image = Image.open(image)
    
    process_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
  
    np_image = process_image(pil_image)
    
    return np_image