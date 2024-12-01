import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os, argparse, time, logging
import model
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm
from torchvision.utils import save_image
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import cv2

def concatenate_images(job_dir, current_epoch, output_path, val_every=5, max_checkpoints=5):
    job_dir = os.path.abspath(job_dir)
    images = []

    for epoch in range(current_epoch, current_epoch - val_every * max_checkpoints, -val_every):
        checkpoint_dir = os.path.join(job_dir, f"epoch={epoch}")
        image_path = os.path.join(checkpoint_dir, "2.png")

        if os.path.exists(image_path):
            image = Image.open(image_path)

            draw = ImageDraw.Draw(image)
            font_size = max(20, image.width // 20)  # Scale font size based on image width
            try:
                font = ImageFont.truetype("LiberationMono-Regular.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()  # Use default font if ttf file is unavailable

            text = f"epoch={epoch}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            text_x = (image.width - text_width) // 2  # Center horizontally
            text_y = 10  # Add some padding from the top

            draw.text((text_x, text_y), text, fill="white", font=font)

            images.append(image)

    if len(images) == 0:
        print("No images found from the last checkpoints.")
        return

    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)
    final_image = Image.new("RGB", (max_width, total_height))

    y_offset = 0
    for img in images:
        final_image.paste(img, (0, y_offset))
        y_offset += img.height

    final_image.save(output_path)

def get_mean_std(data_path, class_name):
    dataset = datasets.ImageFolder(data_path, transform=transforms.ToTensor())
    class_index = dataset.class_to_idx[class_name]
    dataset.samples = [sample for sample in dataset.samples if sample[1] == class_index]

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean = np.zeros(3)
    std = np.zeros(3)
    for images, _ in loader:
        mean += images.mean(dim=(0, 2, 3)).numpy()
        std += images.std(dim=(0, 2, 3)).numpy()
    mean /= len(loader)
    std /= len(loader)
    return mean, std

def compute_validation_loss(model, validation_loader, scheduler, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, _ in validation_loader:
            images = images.to(device)
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (images.shape[0],), device=device).long()

            noisy_images = scheduler.add_noise(images, noise, timesteps)
            predicted_noise = model(noisy_images, timesteps)["sample"]

            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            val_loss += loss.item()

    return val_loss / len(validation_loader)

def save_equalized(checkpoint_path):
    image = cv2.imread(os.path.join(checkpoint_path, "1.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for visualization

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)
    equalized_v = cv2.equalizeHist(v)
    equalized_hsv = cv2.merge((h, s, equalized_v))

    equalized_image = cv2.cvtColor(equalized_hsv, cv2.COLOR_HSV2RGB)
    equalized_image_bgr = cv2.cvtColor(equalized_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(checkpoint_path, "2.png"), equalized_image_bgr)

def moving_average(data, window_size):
    if not 0 < window_size <= len(data):
        raise ValueError("Window size must be between 1 and the length of the data.")
    return [sum(data[i:i + window_size]) / window_size for i in range(len(data) - window_size + 1)]

parser = argparse.ArgumentParser()
parser.add_argument('--jobid', type=str, required=False)
args = parser.parse_args()
jobid = args.jobid

img_size = 224

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logging.info(device)

model = UNet2DModel(
    sample_size=img_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(64, 128, 256, 512),#, 1024),  # More channels -> more parameters
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",  
    ),
)

model = model.to(device)
mean, std = get_mean_std('data/train', 'RUFOUS TREPE')
print(f"mean: {mean}\nstd: {std}")

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),    
    transforms.RandomRotation(degrees=(-10, 10)),             
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

dataset = datasets.ImageFolder('data/train', transform=transform)
abbotts_babbler_index = dataset.class_to_idx['RUFOUS TREPE']
dataset.samples = [sample for sample in dataset.samples if sample[1] == abbotts_babbler_index]

train_size = int(0.9 * len(dataset)) 
val_size = len(dataset) - train_size 

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

print(f"Number of samples in RUFOUS TREPE: {len(dataset)}")
job_dir = f"models/diffusion_jobid={jobid}"
os.makedirs(job_dir, exist_ok=True)
epoch_losses, val_losses = [], []

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = DDPMScheduler(num_train_timesteps=1000)
epochs = 2000
val_every = 50
window_size = 10

for epoch in range(epochs):
    model.train()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    epoch_loss = 0.0  # To accumulate loss for the current epoch
    num_batches = 0
    
    for batch in progress_bar:
        images, _ = batch
        images = images.to(device)

        noise = torch.randn_like(images)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (images.shape[0],), device=device).long()

        noisy_images = scheduler.add_noise(images, noise, timesteps)

        predicted_noise = model(noisy_images, timesteps)["sample"]
        
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        epoch_loss += loss.item()
        num_batches += 1
        optimizer.step()

        progress_bar.set_postfix({"loss": loss.item()})
        
    epoch_loss /= num_batches
    epoch_losses.append(epoch_loss)
    val_losses.append(compute_validation_loss(model, val_loader, scheduler, device))
    
    if (epoch+1) % val_every == 0:
        checkpoint_dir = os.path.join(job_dir, f"epoch={epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"weights.pth"))
        
        model.eval()
        sample = torch.randn((5, 3, img_size, img_size), device=device)  
        for t in scheduler.timesteps:
            with torch.no_grad():
                predicted_noise = model(sample, t)["sample"]
            sample = scheduler.step(predicted_noise, t, sample).prev_sample


        save_image(sample, os.path.join(checkpoint_dir, "1.png"))
        save_equalized(checkpoint_dir)
        output_path = os.path.join(checkpoint_dir, "concatenated_image.png")
        concatenate_images(job_dir, epoch+1, output_path, val_every=val_every)
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Linear scale plot
        axs[0].plot(moving_average(epoch_losses, window_size), label="Training Loss", color="blue", linestyle="-")
        axs[0].plot(moving_average(val_losses, window_size), label="Validation Loss", color="orange", linestyle="--")
        axs[0].set_xlim(0, epochs)  # Set x-axis range
        axs[0].set_ylim(0, max(max(epoch_losses), max(val_losses)) * 1.1)  # Dynamically adjust y-axis
        axs[0].set_title("Loss over Epochs")
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Loss")
        axs[0].legend(loc="upper right")  # Add legend

        # Logarithmic scale plot
        axs[1].plot(moving_average(epoch_losses, window_size), label="Training Loss", color="blue", linestyle="-")
        axs[1].plot(moving_average(val_losses, window_size), label="Validation Loss", color="orange", linestyle="--")
        axs[1].set_xlim(0, epochs)  # Set x-axis range
        axs[1].set_yscale("log")  # Set y-axis to logarithmic scale
        axs[1].set_title("Loss over Epochs (Logarithmic Scale)")
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Loss (log scale)")
        axs[1].legend(loc="upper right")  # Add legend

        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, "loss_plots.png")) 

