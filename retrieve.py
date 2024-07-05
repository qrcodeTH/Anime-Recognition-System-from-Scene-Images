import torch
from torchvision import transforms
from utils import single_picture, show_retrieval_images
from model.model import MyResNeXt101

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

query_path = '' # Your query image path
query_feats = single_picture(model, query_path, transform)
dist_matrix = torch.cdist(query_feats.unsqueeze(0), features.unsqueeze(0)).squeeze()
retrieval_num = 5  # The number of retrieval images you want to show
show_retrieval_images(query_path, dist_matrix, pathlist, label_list, retrieval_num)
