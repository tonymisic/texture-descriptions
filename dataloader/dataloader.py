import torch, os, json
from PIL import Image
from torchvision import transforms

class DTD2(torch.utils.data.Dataset):
    def __init__(self, root_dir, descriptions, splits, split=None):
        self.root_dir = root_dir
        self.split = split
        self.splits = json.load(open(os.path.join(root_dir, splits))) # json file
        self.descriptions = json.load(open(os.path.join(root_dir, descriptions))) # json file
        self.transform = None
    
    def __getitem__(self, index): # return (augmented_image, location_image)
        if self.split == None:
            return None
        elif self.split == 'train':
            return self.read_image(self.splits['train'][index], True)
        elif self.split == 'test':
            return self.read_image(self.splits['test'][index], False)
        elif self.split == 'val':
            return self.read_image(self.splits['val'][index], False)

    def __len__(self): # returns length of current split.
        if self.split == None:
            return len(self.splits['train']) + len(self.splits['test']) + len(self.splits['val'])
        elif self.split == 'train':
            return len(self.splits['train'])
        elif self.split == 'test':
            return len(self.splits['test'])
        elif self.split == 'val':
            return len(self.splits['val'])
    
    def read_image(self, image_location, is_train): # open and transform image, return image and location
        img = Image.open(os.path.join(self.root_dir, image_location)).convert('RGB')
        self.transform = self.transformations(training=is_train)
        # change to give labels

        
        return self.transform(img), 

    def transformations(self, training=True): # standard image transforms from DTD2
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if training:
            return transforms.Compose([ transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        else:
            return transforms.Compose([ transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
