import torch.nn as nn
import torchvision.models as models
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import numpy as np

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

def show_retrieval_images(query_path, dist_matrix, gallery_dirs, label_list, retrieval_num=5):
    query_image = Image.open(query_path).convert('RGB').resize((224, 224), resample=Image.BILINEAR)
    plt.imshow(query_image)
    plt.axis('off')
    plt.title('Query Image')
    plt.show()

    retrieved_labels = set()  # To keep track of retrieved labels
    retrieved_count = 0

    # Sort indices based on distances
    sorted_indices = torch.argsort(dist_matrix)

    for idx in sorted_indices:
        retrieval_dist = dist_matrix[idx].item()
        retrieval_label = label_list[idx]

        # Check if the label has already been retrieved
        if retrieval_label not in retrieved_labels:
            retrieved_labels.add(retrieval_label)
            retrieved_count += 1

            retrieval_image = Image.open(gallery_dirs[idx]).convert('RGB').resize((224, 224), resample=Image.BILINEAR)
            
            # Show the retrieved image
            plt.imshow(retrieval_image)
            plt.axis('off')
            plt.title('Label: {}, Distance: {:.4f}'.format(retrieval_label, retrieval_dist))
            plt.show()

        if retrieved_count == retrieval_num:
            break

# Load your pre-trained model
model = MyResNeXt101()
model.load_state_dict(torch.load('./Resnext101.pth')) #put the resnext101 model path
model = model.cuda()
model = model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load previously saved features and labels
data = torch.load('./features_and_labels.pth') # put your featureandlabel.pth path here
features = data['features']
label_list = data['labels']
pathlist = data['paths']  

query_path = '' # Your query image path
query_feats = single_picture(model, query_path, transform)
dist_matrix = torch.cdist(query_feats.unsqueeze(0), features.unsqueeze(0)).squeeze()
retrieval_num = 5  # The number of retrieval images you want to show
show_retrieval_images(query_path, dist_matrix, pathlist, label_list, retrieval_num)
