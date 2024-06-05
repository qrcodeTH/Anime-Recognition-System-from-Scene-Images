import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import MyDataset, getFileList, extract_features
from model.model import MyResNeXt101

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
path = '/kaggle/working/Train'
pathlist = getFileList(path)
labels = [p.split('/')[-2] for p in pathlist]
dataset = MyDataset(dirs=pathlist, labels=labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
features, label_list = extract_features(model, dataloader)

# Save features, labels, and image paths
data = {
    'features': features,
    'labels': label_list,
    'paths': pathlist
}
torch.save(data, './features_and_labels.pth')
