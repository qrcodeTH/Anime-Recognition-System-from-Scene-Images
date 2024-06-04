import torch.nn as nn
import torchvision.models as models
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

class MyResNeXt101(nn.Module):
    def __init__(self, num_classes=100):
        super(MyResNeXt101, self).__init__()
        self.network = models.resnext101_32x8d(pretrained=True)

        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.network(x)

class MyDataset(Dataset):
    def __init__(self, dirs, labels, transform=None):
        self.dirs = dirs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        img_path = self.dirs[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def getFileList(root):
    file_list = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_list.append(os.path.join(path, name))
    return file_list

def extract_features(model, dataloader):
    features = torch.FloatTensor()
    label_list = []
    for img, label in tqdm(dataloader):
        img = img.cuda()
        outputs = model(img)
        ff = outputs.data.cpu()
        features = torch.cat((features, ff), 0)
        label_list += list(label)
    return features, label_list

def single_picture(model, query_path, transform):
    img = Image.open(query_path)
    img = transform(img)
    img = img.cuda()
    img = img.unsqueeze(0)
    outputs = model(img)
    outputs = outputs.data.cpu()
    return outputs
            
# Load your pre-trained model
model = MyResNeXt101()
model.load_state_dict(torch.load('./Resnext101.pth')) # put resnext101 model path here
model.classifier = nn.Sequential()
model = model.cuda()
model = model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset and extract features
path = './Train'
pathlist = getFileList(path)
labels = [p.split('/')[-2] for p in pathlist]
dataset = MyDataset(dirs=pathlist, labels=labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
features, label_list = extract_features(model, dataloader)

# Save features and labels
torch.save({'features': features, 'labels': label_list}, 'Retrieval_Resnext101_features_and_labels.pth')