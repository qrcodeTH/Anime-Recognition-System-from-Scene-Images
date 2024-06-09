import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
from utils import MyDataset, getFileList, extract_features
from model.model import MyResNeXt101

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

# Load saved features and other necessary data
features = torch.load('./features.pth')
pathlist = torch.load('./pathlist.pth')
label_list = torch.load('./label_list.pth')

# Load test dataset
test_path = './dataset/Test'
test_pathlist = getFileList(test_path)
test_labels = [p.split('/')[-2] for p in test_pathlist]
test_dataset = MyDataset(dirs=test_pathlist, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# Define your evaluation functions
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

        top_k_labels = []
        for idx in retrieval_indices:
            label = label_list[idx]
            if label not in top_k_labels:
                top_k_labels.append(label)
            if len(top_k_labels) == top_k:
                break

        if query_label[0] in top_k_labels:
            top5_correct += 1

    top1_accuracy = top1_correct / total_images
    top5_accuracy = top5_correct / total_images

    return top1_accuracy, top5_accuracy

def evaluate_lbap_mrr(model, dataloader, pathlist, label_list):
    features_cuda = features.cuda()
    lbaps = []
    reciprocal_ranks = []

    for query_img, query_label in tqdm(dataloader):
        query_img = query_img.cuda()

        with torch.no_grad():
            query_feats = model(query_img)
            dist_matrix = torch.cdist(query_feats.unsqueeze(0), features_cuda.unsqueeze(0)).squeeze()
            retrieval_indices = torch.argsort(dist_matrix)

        true_labels = [1 if label_list[idx] == query_label[0] else 0 for idx in retrieval_indices]
        scores = -dist_matrix[retrieval_indices].detach().cpu().numpy()  # Invert distance to get scores

        unique_labels = []
        unique_scores = []
        for label, score in zip(true_labels, scores):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_scores.append(score)

        if sum(true_labels) > 0:  
            lbap = label_ranking_average_precision_score([true_labels], [scores])
            lbaps.append(lbap)
        for rank, idx in enumerate(retrieval_indices):
            if label_list[idx] == query_label[0]:
                reciprocal_ranks.append(1.0 / (rank + 1))
                break
        else:
            reciprocal_ranks.append(0)  

    # Compute mean of the LBAPs and reciprocal ranks
    mean_lbap = np.mean(lbaps)
    mean_rr = np.mean(reciprocal_ranks)

    return mean_lbap, mean_rr

# Define and load the test dataset and dataloader
test_dataset = MyDataset(dirs=pathlist, labels=label_list, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

# Evaluate accuracy
top1_acc, top5_acc = evaluate_accuracy(model, test_dataloader, pathlist, label_list)

# Evaluate LBAP and MRR
mean_lbap, mean_rr = evaluate_lbap_mrr(model, test_dataloader, pathlist, label_list)

print(f"Top-1 Accuracy: {top1_acc * 100:.2f}%")
print(f"Top-5 Accuracy: {top5_acc * 100:.2f}%")
print(f"Mean LBAP (Label-Based Average Precision): {mean_lbap * 100:.2f}%")
print(f"Mean Reciprocal Rank (MRR): {mean_rr * 100:.2f}%")
