import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch import optim
from loss.labelloss import LabelSmoothing  
import torch.nn as nn
import torchvision.models as models

class MyResNeXt101(nn.Module):
    def __init__(self, num_classes=100):
        super(MyResNeXt101, self).__init__()
        self.network = models.resnext101_32x8d(pretrained=True)

        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.network(x)
    
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Function for training and validation
def train_val(start_epoch=0, resume_path=None):
    model.train()
    best_acc = 0.0
    checkpoint_files = []

    if resume_path:
        checkpoint = torch.load(resume_path)

        if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f'Resuming training from epoch {start_epoch}')
        else:
            model.load_state_dict(checkpoint)
            start_epoch = 1  
            print(f'Resuming training from epoch {start_epoch} with direct model state loading.')
    else:
        print('Starting training from scratch')

    for epoch in range(start_epoch, epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_loss = 0.0
        train_count = 0
        model.train()
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_count += torch.sum(labels == preds)
        train_loss /= len(trainset)
        train_acc = train_count.double() / len(trainset)

        print(f'Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}')

        val_loss = 0.0
        val_count = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(valloader):
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_count += torch.sum(labels == preds)
            val_loss /= len(valset)
            val_acc = val_count.double() / len(valset)
            
            print(f'Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}')
            
            if best_acc < val_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), '/kaggle/working/best_model.pth')
                print(f'Saving best model with accuracy: {val_acc:.4f}')

            # Save checkpoint
            checkpoint_filename = f'/kaggle/working/checkpoint_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_filename)

            checkpoint_files.append(checkpoint_filename)
            if len(checkpoint_files) > 3:
                old_checkpoint = checkpoint_files.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)

            with open('/kaggle/working/training_log.txt', 'a') as f:
                f.write(f'Epoch {epoch + 1}/{epochs}\n')
                f.write(f'Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}\n')
                f.write(f'Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}\n\n')

# Parameters
lr = 0.01
epochs = 100
smoothing = 0.1
batch_size = 32
trainpath = './Train'
valpath = './Validation'
resume_path = None  # Set to None to start training from scratch

# Data transformations 
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets and create data loaders
trainset = datasets.ImageFolder(trainpath, transform=train_transforms)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

valset = datasets.ImageFolder(valpath, transform=val_transforms)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

model = MyResNeXt101(num_classes=100).cuda()
criterion = LabelSmoothing(smoothing=smoothing).cuda()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0, nesterov=False)

# Start training and validation
train_val(start_epoch=0, resume_path=resume_path)

    