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

# Import remaining functions from main.py
from main import reciprocal_rank, mean_reciprocal_rank, accuracy, top_5_accuracy

# Calculate the MRR score
mrr_score = mean_reciprocal_rank(model, test_loader)

# Calculate accuracy score
accuracy_score = accuracy(model, test_loader)

# Calculate top-5 accuracy
top_5_acc = top_5_accuracy(model, test_loader)

print(f"Accuracy: {accuracy_score:.5%}")
print(f"Top-5 Accuracy: {top_5_acc:.5%}")
print(f"Mean Reciprocal Rank (MRR): {mrr_score:.5%}")
print(f"Mean Average Precision (MAP): {mean_lraps:.5%}")
