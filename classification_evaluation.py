import torch
from torchvision import transforms, datasets
from tqdm import tqdm  
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
from main import train_val

# Import the model classes
from model.model import MyResNeXt101

# Define test transformations
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the saved model
model = MyResNeXt101(num_classes=100).cuda()

# Load the state dictionary
state_dict = torch.load('./best_model.pth')  # Path to the model
model.load_state_dict(state_dict)

# Load the test dataset
test_dir = "./Test"  # Path to the test set
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

label_map = {v: k for k, v in test_dataset.class_to_idx.items()}  

model.eval()
y_true_list = []
y_score_list = []

for images, labels in test_loader:
    images = images.cuda()  
    labels = labels.cuda()
    
    outputs = model(images)
    
    probabilities = torch.softmax(outputs, dim=1)

    labels_np = labels.cpu().numpy()
    probabilities_np = probabilities.detach().cpu().numpy()

    y_true_list.append(labels_np)
    y_score_list.append(probabilities_np)

y_true = np.concatenate(y_true_list, axis=0)
y_score = np.concatenate(y_score_list, axis=0)

y_true_binary = np.zeros_like(y_score)
for i, sample in enumerate(y_true):
    y_true_binary[i, sample] = 1

# Calculate the label ranking average precision score
mean_lraps = label_ranking_average_precision_score(y_true_binary, y_score, sample_weight=None)

# Function to calculate the reciprocal rank
def reciprocal_rank(output, label):
    _, top_5_indices = output.topk(5, 1, True, True)
    top_5_indices = top_5_indices.cpu().numpy()
    label = label.cpu().numpy()
    
    for rank, idx in enumerate(top_5_indices[0]):
        if idx == label[0]:
            return 1.0 / (rank + 1)
    return 0.0

# Function to calculate MRR
def mean_reciprocal_rank(model, data_loader):
    reciprocal_ranks = []

    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            rr = reciprocal_rank(outputs, labels)
            reciprocal_ranks.append(rr)

    return np.mean(reciprocal_ranks)

# Calculate the MRR score
mrr_score = mean_reciprocal_rank(model, test_loader)

# Function to calculate accuracy
def accuracy(model, data_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total

# Function to calculate top-5 accuracy
def top_5_accuracy(model, data_loader):
    correct_top_5 = 0
    total = 0

    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            _, top_5 = outputs.topk(5, 1, True, True)

            # Check if the correct label is in the top-5 predictions
            correct_top_5 += torch.sum(top_5 == labels.unsqueeze(1)).item()
            total += labels.size(0)

    return correct_top_5 / total

# Calculate accuracy score
accuracy_score = accuracy(model, test_loader)

# Calculate top-5 accuracy
top_5_acc = top_5_accuracy(model, test_loader)

print(f"Accuracy: {accuracy_score:.5%}")
print(f"Top-5 Accuracy: {top_5_acc:.5%}")
print(f"Mean Reciprocal Rank (MRR): {mrr_score:.5%}")
print(f"Mean Average Precision (MAP): {mean_lraps:.5%}")
