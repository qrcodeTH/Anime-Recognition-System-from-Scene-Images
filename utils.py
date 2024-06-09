import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

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
    # Ensure the image has 3 channels
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = transform(img)
    img = img.cuda()  # Move image to GPU
    img = img.unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    return output

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
