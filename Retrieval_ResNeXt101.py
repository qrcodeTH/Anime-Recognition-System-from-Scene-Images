import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import faiss
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
class MyResNeXt101(nn.Module):
    def __init__(self, num_classes=100):
        super(MyResNeXt101, self).__init__()
        self.network = models.resnext101_32x8d(pretrained=True)

        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.network(x)

# Load the pre-trained model
model_path = "" # Replace with model from classification
model = MyResNeXt101(num_classes=100)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom dataset to load images and their labels
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}

        label_dirs = sorted(os.listdir(root_dir))
        for idx, label_dir in enumerate(label_dirs):
            self.label_to_idx[label_dir] = idx
            self.idx_to_label[idx] = label_dir
            label_path = os.path.join(root_dir, label_dir)
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                self.image_paths.append(image_path)
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, image_path

# Load the database images
database_dir = "./Train"
database_dataset = ImageDataset(database_dir, transform)
database_loader = DataLoader(database_dataset, batch_size=256, shuffle=False, num_workers=4)

# Extract features from the database images
features = []
labels = []
image_paths = []

with torch.no_grad():
    for images, batch_labels, batch_image_paths in database_loader:
        images = images.to(device)
        outputs = model(images).cpu()
        features.append(outputs)
        labels.extend(batch_labels.numpy())
        image_paths.extend(batch_image_paths)

features = torch.cat(features).numpy()

# Convert features to float32 (required by FAISS)
features = features.astype('float32')

# Build index
index = faiss.IndexFlatL2(features.shape[1])
index.add(features)

def retrieve_with_faiss(query_image_path, top_k=5):
    query_image = Image.open(query_image_path).convert('RGB')
    query_image = transform(query_image).unsqueeze(0).to(device)

    with torch.no_grad():
        query_feature = model(query_image).cpu().numpy()

    query_feature = query_feature.astype('float32')

    # Use FAISS to find similar images
    _, top_k_indices = index.search(query_feature, top_k)

    top_k_labels = [database_dataset.idx_to_label[labels[idx]] for idx in top_k_indices[0]]
    top_k_images = [image_paths[idx] for idx in top_k_indices[0]]

    return top_k_labels, top_k_images