import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import os, argparse, time, logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

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
    
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_p=0.2, negative_slope=0.01):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         self.batchnorm = nn.BatchNorm2d(out_channels)
#         self.dropout = nn.Dropout(dropout_p)

#         self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

#         self.match_dimensions = (in_channels != out_channels or stride != 1)
#         if self.match_dimensions:
#             self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

#     def forward(self, x):
#         residual = x 

#         out = self.conv(x)
#         out = self.batchnorm(out)
#         out = self.leaky_relu(out)  
#         out = self.dropout(out)

#         if self.match_dimensions:
#             residual = self.residual_conv(residual)

#         out += residual
#         out = self.leaky_relu(out)  

#         return out
        
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = ConvBlock(3, 32, stride=2)
        self.conv2 = ConvBlock(32, 64, stride=2)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = ConvBlock(64, 128, stride=2)
        self.conv5 = ConvBlock(128, 128)
        self.conv6 = ConvBlock(128, 128)
        self.conv7 = ConvBlock(128, 256, stride=2)
        self.conv8 = ConvBlock(256, 256)
        self.conv9 = ConvBlock(256, 256)
        self.conv10 = ConvBlock(256, 256)
        
        self.fc1 = nn.Linear(256 * 14 * 14, 256)
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
        x = self.conv9(x)
        x = self.conv10(x)
        # x = self.conv11(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.bn(self.fc1(x))) 
        x = self.final_dropout(x)
        
        x = self.fc2(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
logger.info(f'Device used: {device}')
model = SimpleCNN()
print(model)
model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Adam optimizer

transform = transforms.Compose([
    transforms.Resize((224, 224)),                       
    transforms.RandomHorizontalFlip(p=0.5),              
    transforms.RandomVerticalFlip(p=0.1),    
    # transforms.RandomRotation(degrees=15),               
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), 
    # transforms.RandomInvert(p=0.1),
    transforms.RandomGrayscale(p=0.1),                   
    # transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),  
    transforms.RandomAutocontrast(p=0.3),              
    # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   # Normalize using ImageNet statistics
])

dataset = datasets.ImageFolder('data/train', transform=transform)
train_size = int(0.85 * len(dataset)) 
val_size = len(dataset) - train_size 

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
batch_size = 7000
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

parser = argparse.ArgumentParser()
parser.add_argument('--jobid', type=str, required=True)
args = parser.parse_args()

num_epochs = 200
val_interval = 10
jobid = args.jobid
checkpoint_dir = f"models/classifier_jobid={jobid}"
os.makedirs(checkpoint_dir, exist_ok=True)

# TEST

# val_dataset =  datasets.ImageFolder('data/valid', transform=transform)
# val_loader = DataLoader(val_dataset, batch_size=2000, shuffle=False)

# model = SimpleCNN()
# model = model.to(device)

# # state_dict = torch.load("model_weights/classifier_weights.pth", map_location=torch.device('cpu'), weights_only=True)
# state_dict = torch.load("models/classifier_jobid=44459/checkpoint_step_1230.pth", map_location=torch.device('cpu'), weights_only=True)
# new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# model.load_state_dict(new_state_dict)
# model.eval()

# with torch.no_grad():
#     val_loss = 0
#     correct = 0
#     total = 0
#     for val_images, val_labels in val_loader:
#         val_images, val_labels = val_images.to(device), val_labels.to(device)
#         val_outputs = model(val_images)
#         val_loss += criterion(val_outputs, val_labels).item()
#         _, predicted = torch.max(val_outputs, 1)
#         correct += (predicted == val_labels).sum().item()
#         total += val_labels.size(0)
#         print('labels:\t' + str(val_labels))
#         print('predicted:\t' + str(predicted))
        
#         labels_df = pd.read_csv('labels.csv')
#         class_to_label = dict(zip(labels_df['class_id'], labels_df['label']))
#         print([class_to_label.get(predicted[i].item(), 'Unknown Species') for i in range(2000)])
#         break
    
#     avg_val_loss = val_loss #/ len(val_loader)
#     accuracy = 100 * correct / total
#     print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

# END TEST

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
                
                avg_val_loss = val_loss / len(val_loader)
                accuracy = 100 * correct / total
                logger.info(f'Step [{step}], Training Loss: {loss.item():.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
            
            model.train()
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time  # Calculate duration
    logger.info(f'Epoch [{epoch + 1}/{num_epochs}] Complete, Loss: {loss.item():.4f}, Time: {epoch_duration:.2f} seconds')