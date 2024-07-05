import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
from utils import MyDataset, getFileList, extract_features
from model.model import MyResNeXt101
import os

# Load your pre-trained model
model = MyResNeXt101()
model.load_state_dict(torch.load('./Resnext101.pth')) # Put the resnext101 model path here
model = model.cuda()
model = model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Load dataset and extract features
path = '/kaggle/working/Train'
pathlist = getFileList(path)
labels = [p.split('/')[-2] for p in pathlist]
dataset = MyDataset(dirs=pathlist, labels=labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
features, label_list = extract_features(model, dataloader)


# Load the test dataset
test_dir = '/kaggle/working/Test'
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Create label_map
label_map = {v: k for k, v in test_dataset.class_to_idx.items()}

# Initialize y_true as an empty list
y_true = []

# Iterate through dataset and construct one-hot encoded labels
for img_path, label in test_dataset.imgs:
    # Extract true label from image path
    true_label = os.path.basename(os.path.dirname(img_path))
    true_label_index = label_map[label]  # Get the index from label_map
    
    # Convert true_label_index to integer if needed
    if isinstance(true_label_index, str):
        # Handle cases where true_label_index is the anime title instead of an index
        true_label_index = [k for k, v in label_map.items() if v == true_label][0]
    
    # Create a one-hot encoded label array
    one_hot_label = np.zeros(len(label_map), dtype=np.float32)
    one_hot_label[true_label_index] = 1
    
    # Append the one-hot encoded label to y_true
    y_true.append(one_hot_label)

# Convert y_true to numpy array
y_true = np.array(y_true)

def retrieve_most_similar_images(query_path, model, features, label_list, gallery_dirs, transform):
    # Prepare the query image
    query_features = single_picture(model, query_path, transform)
    
    # Ensure query_features is on the same device as img_features
    device = features[0].device
    query_features = query_features.to(device)
    
    # Initialize a dictionary to store the closest image paths, distances, and labels
    closest_images = {}
    
    # Loop over the dataset features to find the closest images
    for idx, (img_features, label) in enumerate(zip(features, label_list)):
        # Calculate L2 distance between query and current image features
        distance = torch.norm(query_features - img_features.unsqueeze(0), p=2).item()
        
        # Update closest image if current image is closer or first entry for the label
        if label not in closest_images or distance < closest_images[label]['distance']:
            closest_images[label] = {'distance': distance, 'path': gallery_dirs[idx]}
    
    # Convert closest_images dictionary to a list
    closest_images_list = [{'label': label, 'distance': info['distance'], 'path': info['path']} for label, info in closest_images.items()]
    
    # Sort the closest images by distance within each label
    closest_images_list.sort(key=lambda x: x['distance'])
    
    return closest_images_list

def create_distance_array(closest_images, label_map):
    # Initialize an array to store distances
    num_labels = len(label_map)
    distance_array = [-1] * num_labels  # Initialize with -1
    
    # Fill the array with distances where available
    for image_info in closest_images:
        label = image_info['label']
        distance = image_info['distance']
        if label in label_map.values():
            index = list(label_map.keys())[list(label_map.values()).index(label)]
            distance_array[index] = distance
    
    return distance_array

def display_distances_and_labels(closest_images):
    # Display distances and labels sorted by distance for each label
    for image_info in closest_images:
        distance = image_info['distance']
        label = image_info['label']
        print(f"Label: {label}, Distance: {distance:.4f}")

# Example usage
query_path = '/kaggle/working/Test/Shingeki no Kyojin/79f43bc0-3671-4ff6-89ce-e75315788ef5.jpg'

# Retrieve closest images
closest_images = retrieve_most_similar_images(query_path, model, features, label_list, dataset.dirs, transform)

# Display distances and labels sorted by distance for each label
display_distances_and_labels(closest_images)

query_dir = '/kaggle/working/Test'
all_distances = []

# Use tqdm to wrap the outer loop to show progress
for label in tqdm(sorted(os.listdir(query_dir)), desc='Processing labels'):
    label_dir = os.path.join(query_dir, label)
    if os.path.isdir(label_dir):
        for image_file in tqdm(sorted(os.listdir(label_dir)), desc=f'Processing images in {label}'):
            image_path = os.path.join(label_dir, image_file)
            closest_images = retrieve_most_similar_images(image_path, model, features, label_list, dataset.dirs, transform)
            distance_array = create_distance_array(closest_images, label_map)
            all_distances.append(distance_array)

# Combine all_distances into y_score
y_score = []

# Append each distance list from all_distances to y_score
for distances in all_distances:
    y_score.append(distances)

def distances_to_probabilities(y_score):
    # Convert the list of lists to a NumPy array for easier manipulation
    y_score_array = np.array(y_score)
    
    # Convert distances to negative values
    negative_distances = -y_score_array
    
    # Apply the softmax function to the negative distances
    probabilities = F.softmax(torch.tensor(negative_distances), dim=1).numpy()
    
    return probabilities

# Transform distance scores to probabilities
y_score_probabilities = distances_to_probabilities(y_score)

# Evaluate LRAP score
lrap_score = label_ranking_average_precision_score(y_true, y_score_probabilities)

def mean_reciprocal_rank_from_distances(y_true, y_score_distances):
    # Calculate reciprocal ranks
    reciprocal_ranks = np.zeros_like(y_true, dtype=np.float32)
    
    for i in range(len(y_true)):
        # Sort distances in ascending order
        sorted_indices = np.argsort(y_score_distances[i])
        
        # Find the rank of true class (where y_true[i] == 1)
        true_class_rank = np.where(sorted_indices == np.argmax(y_true[i]))[0][0]
        
        # Calculate reciprocal rank
        reciprocal_ranks[i] = 1.0 / (true_class_rank + 1)
    
    # Calculate MRR
    mrr = np.mean(reciprocal_ranks)
    
    return mrr

# Evaluate MRR
mrr = mean_reciprocal_rank_from_distances(y_true, y_score)

# Function to evaluate accuracy
def evaluate_accuracy(model, dataloader, pathlist, label_list, top_k=5):
    total_images = 0
    top1_correct = 0
    top5_correct = 0

    # Move features to GPU
    features_cuda = features.cuda()

    for query_img, query_label in tqdm(dataloader):
        total_images += 1
        query_img = query_img.cuda()

        # Perform retrieval using the query image
        query_feats = model(query_img)
        dist_matrix = torch.cdist(query_feats.unsqueeze(0), features_cuda.unsqueeze(0)).squeeze()
        retrieval_indices = torch.argsort(dist_matrix)

        # Check top-1 accuracy
        if label_list[retrieval_indices[0]] == query_label[0]:
            top1_correct += 1

        # Check top-k accuracy
        top_k_labels = [label_list[idx] for idx in retrieval_indices[:top_k]]
        if query_label[0] in top_k_labels:
            top5_correct += 1

    top1_accuracy = top1_correct / total_images
    top5_accuracy = top5_correct / total_images

    return top1_accuracy, top5_accuracy

# Evaluate accuracy
top1_acc, top5_acc = evaluate_accuracy(model, test_dataloader, pathlist, label_list)


print(f"Top-1 Accuracy: {top1_acc * 1:.4f}")
print(f"Top-5 Accuracy: {top5_acc * 1:.4f}")
print(f"LRAP score: {lrap_score}")
print(f"Mean Reciprocal Rank (MRR): {mrr}")
