# src/cloth_segmentation.py
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import cv2
import sys

# Add this line to make sure that the root directory can be resolved.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from config import IMAGE_HEIGHT, IMAGE_WIDTH
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch.nn as nn

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

# --- Dataset and DataLoader (Using your Labeled Data)---
# Define default transforms for images and masks
image_transform_default = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform_default = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0], std=[1.0])  # No-op normalization for mask
])

class ClothDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir  # Directory containing the masks.
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]  # Assuming images are png or jpg.
        self.transform = transform if transform is not None else image_transform_default
        self.mask_transform = mask_transform if mask_transform is not None else mask_transform_default

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name) # Assuming masks have the same name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask

        image = self.transform(image)
        mask = self.mask_transform(mask)

        # Convert mask to class indices
        mask = (mask > 0.5).float()
        mask = mask.long()

        return image, mask

# --- Model and Training ---
def create_model(num_classes=2):  # num_classes: Background (0), Cloth (1)
    """
    Creates a pre-trained DeepLabV3+ model, modifies the classifier, and sets it to training mode.
    """
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
    # Modify the classifier to match the number of classes
    model.classifier = DeepLabHead(2048, num_classes)    # Set model to training mode
    model.train()
    return model

def train_epoch(model, dataloader, optimizer, device):
    """Trains the model for one epoch."""
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs['out'], masks.squeeze(1).long()) # cross entropy loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch Loss: {epoch_loss:.4f}")
    return epoch_loss
def evaluate_model(model, dataloader, device):
    """Evaluates the model."""
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs['out'], masks.squeeze(1).long())
            running_loss += loss.item()
    epoch_loss = running_loss / len(dataloader)
    print(f"Validation Loss: {epoch_loss:.4f}")
    return epoch_loss

def cloth_segmentation_training(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, num_epochs=10, learning_rate=0.001, batch_size=4):
    """
    Fine-tunes a DeepLabV3+ model for cloth segmentation.

    Args:
        train_image_dir (str): Path to the directory of training images.
        train_mask_dir (str): Path to the directory of training masks.
        val_image_dir (str): Path to the directory of validation images.
        val_mask_dir (str): Path to the directory of validation masks.
        num_epochs (int): Number of training epochs.
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The batch size for training.
    """
    # --- Device Setup ---
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Preprocessing ---
    # Define the transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        # Normalize with ImageNet stats
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Create datasets and data loaders
    train_dataset = ClothDataset(train_image_dir, train_mask_dir, transform=transform)
    val_dataset = ClothDataset(val_image_dir, val_mask_dir, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model, Optimizer, and Loss Function ---
    model = create_model()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_epoch(model, train_dataloader, optimizer, device)
        evaluate_model(model, val_dataloader, device)

    # --- Save the Trained Model (Optional) ---
    model_save_path = "cloth_segmentation_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")

def cloth_segmentation(image_path, model_path="cloth_segmentation_model.pth"):
    """
    Segments the cloth in an image using a trained DeepLabV3+ model.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained model.

    Returns:
        PIL.Image or None: A PIL image representing the cloth mask, or None if an error occurred.
    """
    try:
        # --- Device Setup ---
        device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

        # --- Load the Trained Model ---
        model = create_model()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)

        # --- Preprocessing ---
        transform = transforms.Compose([
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # --- Load and Preprocess the Image ---
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # --- Make Predictions ---
        with torch.no_grad():
            output = model(input_tensor)['out']
            # --- Post-processing ---
            output_predictions = output.argmax(1)  # Get the predicted class for each pixel
            mask = output_predictions[0].cpu().numpy() # move to cpu and convert to numpy

        # --- Create a Mask Image (Cloth = 1, Background = 0) ---
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        return mask_image

    except Exception as e:
        print(f"Error during cloth segmentation: {e}")
        return None

def visualize_mask(image, mask, alpha=0.5):
    """
    Visualizes the mask overlaid on the original image.

    Args:
        image (PIL.Image): The original image.
        mask (PIL.Image): The segmentation mask (grayscale).
        alpha (float): The transparency of the mask overlay (0 to 1).

    Returns:
        PIL.Image: The image with the mask overlaid.
    """
    try:
        image_np = np.array(image)
        mask_np = np.array(mask) # Convert mask to numpy array
        # Ensure the mask is grayscale (single channel)
        if len(mask_np.shape) == 3:
            mask_np = mask_np[:, :, 0]  # Use the first channel if it's a 3-channel mask
        # Ensure mask is the correct size (resize it if necessary)
        if mask_np.shape[:2] != image_np.shape[:2]:
            mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST) # nearest neighbor
        # Create a color mask
        color_mask = np.zeros_like(image_np)
        color_mask[mask_np > 0] = [255, 0, 255]  # Purple color for cloth

        # Overlay the color mask on the image
        output_image = image_np.copy() # copy
        output_image = cv2.addWeighted(output_image, 1, color_mask, alpha, 0) # Overlay

        return Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

    except Exception as e:
        print(f"Error visualizing mask: {e}")
        return None

# --- Example Usage (for training and testing) ---
if __name__ == '__main__':
    # -----------------------------------
    # 1. Data paths
    # -----------------------------------
    train_image_dir = os.path.join(DATA_DIR, "train", "images")  # Your training image directory
    train_mask_dir = os.path.join(DATA_DIR, "train", "masks")    # Your training mask directory
    val_image_dir = os.path.join(DATA_DIR, "val", "images")      # Your validation image directory
    val_mask_dir = os.path.join(DATA_DIR, "val", "masks")        # Your validation mask directory

    # -----------------------------------
    # 2. Training
    # -----------------------------------
    # Fine-tune the model (replace with your actual training data paths)
    cloth_segmentation_training(
        train_image_dir=train_image_dir,
        train_mask_dir=train_mask_dir,
        val_image_dir=val_image_dir,
        val_mask_dir=val_mask_dir,
        num_epochs=5,  # Adjust as needed
        learning_rate=0.001,
        batch_size=2, # Adjust to fit GPU
    )

    # -----------------------------------
    # 3. Testing
    # -----------------------------------
    test_image_path = os.path.join(DATA_DIR, "val", "images", "00003_00.jpg")  # Path to your test image
    cloth_mask = cloth_segmentation(image_path=test_image_path)

    if cloth_mask:
        # Visualize the mask
        original_image = Image.open(test_image_path).convert("RGB")  # Load the original image.
        visualized_image = visualize_mask(original_image, cloth_mask, alpha=0.5)

        if visualized_image:
            visualized_image.save("cloth_mask_output.png")
            visualized_image.show()
            print("Cloth mask visualized and saved as cloth_mask_output.png")