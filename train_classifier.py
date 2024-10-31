import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os, argparse, time, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_p=0.2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x
        
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = ConvBlock(3, 16, stride=2)
        self.conv2 = ConvBlock(16, 32, stride=2)
        self.conv3 = ConvBlock(32, 32)
        self.conv4 = ConvBlock(32, 64, stride=2)
        self.conv5 = ConvBlock(64, 64)
        self.conv6 = ConvBlock(64, 64)
        self.conv7 = ConvBlock(64, 64)
        self.conv8 = ConvBlock(64, 64)
        
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.final_dropout = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 525)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        
        x = torch.flatten(x, 1)  # Flatten before the fully connected layers
        x = F.relu(self.bn(self.fc1(x))) 
        x = self.final_dropout(x)
        
        x = self.fc2(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f'Device used: {device}')
model = SimpleCNN()
model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder('data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=12288, shuffle=True)

val_dataset =  datasets.ImageFolder('data/valid', transform=transform)
val_loader = DataLoader(train_dataset, batch_size=2500, shuffle=True)

parser = argparse.ArgumentParser()
parser.add_argument('--jobid', type=str, required=True)
args = parser.parse_args()

num_epochs = 100
val_interval = 10
jobid = args.jobid
checkpoint_dir = f"models/classifier_jobid={jobid}"
os.makedirs(checkpoint_dir, exist_ok=True)

step = 0 
for epoch in range(num_epochs):
    epoch_start_time = time.time() 
    for images, labels in train_loader:
        optimizer.zero_grad()
        
        outputs = model(images.to(device))
        loss = criterion(outputs, labels.to(device))
        
        loss.backward()
        optimizer.step()
        
        step += 1
        
        if step % val_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f'Saved checkpoint to {checkpoint_path}')
            
            model.eval() 
            with torch.no_grad():
                val_loss = 0
                correct = 0
                total = 0
                for val_images, val_labels in val_loader:
                    val_images, val_labels = val_images.to(device), val_labels.to(device)
                    val_outputs = model(val_images)
                    val_loss += criterion(val_outputs, val_labels).item()
                    _, predicted = torch.max(val_outputs, 1)
                    correct += (predicted == val_labels).sum().item()
                    total += val_labels.size(0)
                    break  # Ensure you only evaluate on one batch for now
                
                avg_val_loss = val_loss  # / len(val_loader)
                accuracy = 100 * correct / total
                logger.info(f'Step [{step}], Training Loss: {loss.item():.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
            
            model.train()
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time  # Calculate duration
    logger.info(f'Epoch [{epoch + 1}/{num_epochs}] Complete, Loss: {loss.item():.4f}, Time: {epoch_duration:.2f} seconds')