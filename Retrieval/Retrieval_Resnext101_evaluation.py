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

# Load your pre-trained model
model = MyResNeXt101()
model.load_state_dict(torch.load('./Resnext101.pth')) #put the resnext101 model path
model.classifier = nn.Sequential()
model = model.cuda()
model = model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load previously saved features and labels
data = torch.load('') # put resnext101 model path here
features = data['features']
label_list = data['labels']

# Load test dataset
test_path = './dataset/Test'
test_pathlist = getFileList(test_path)
test_labels = [p.split('/')[-2] for p in test_pathlist]
test_dataset = MyDataset(dirs=test_pathlist, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# Function to evaluate accuracy
def evaluate_accuracy(model, dataloader, pathlist, label_list, top_k=5):
    total_images = 0
    top1_correct = 0
    top5_correct = 0

    features_cuda = features.cuda()

    for query_img, query_label in tqdm(dataloader):
        total_images += 1
        query_img = query_img.cuda()

        query_feats = model(query_img)
        dist_matrix = torch.cdist(query_feats.unsqueeze(0), features_cuda.unsqueeze(0)).squeeze()
        retrieval_indices = torch.argsort(dist_matrix)

        if label_list[retrieval_indices[0]] == query_label[0]:
            top1_correct += 1

        top_k_labels = [label_list[idx] for idx in retrieval_indices[:top_k]]
        if query_label[0] in top_k_labels:
            top5_correct += 1

    top1_accuracy = top1_correct / total_images
    top5_accuracy = top5_correct / total_images

    return top1_accuracy, top5_accuracy

# Evaluate accuracy
top1_acc, top5_acc = evaluate_accuracy(model, test_dataloader, test_pathlist, test_labels)

def evaluate_map_mrr(model, dataloader, pathlist, label_list):
    features_cuda = features.cuda()

    average_precisions = []
    reciprocal_ranks = []

    for query_img, query_label in tqdm(dataloader):
        query_img = query_img.cuda()

        with torch.no_grad():
            query_feats = model(query_img)
            dist_matrix = torch.cdist(query_feats.unsqueeze(0), features_cuda.unsqueeze(0)).squeeze()
            retrieval_indices = torch.argsort(dist_matrix)

        true_labels = [1 if label_list[idx] == query_label[0] else 0 for idx in retrieval_indices]
        scores = -dist_matrix[retrieval_indices].detach().cpu().numpy() 

        ap = average_precision_score(true_labels, scores)
        average_precisions.append(ap)

        for rank, idx in enumerate(retrieval_indices):
            if label_list[idx] == query_label[0]:
                reciprocal_ranks.append(1.0 / (rank + 1))
                break
        else:
            reciprocal_ranks.append(0)  

    mean_ap = np.mean(average_precisions)
    mean_rr = np.mean(reciprocal_ranks)

    return mean_ap, mean_rr

mean_ap, mean_rr = evaluate_map_mrr(model, test_dataloader, test_pathlist, test_labels)

print(f"Top-1 Accuracy: {top1_acc * 100:.4f}%")
print(f"Top-5 Accuracy: {top5_acc * 100:.4f}%")
print(f"Mean Average Precision (mAP): {mean_ap * 100:.4f}%")
print(f"Mean Reciprocal Rank (MRR): {mean_rr * 100:.4f}%")
