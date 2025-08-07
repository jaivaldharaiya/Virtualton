# src/human_parsing.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from config import *  # Import configuration

# --- Pre-trained Model Setup (DeepLabV3+) ---
try:
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True) # Or a different model like 'deeplabv3_mobilenet_v3_large'
    model.eval()  # Set the model to evaluation mode
except Exception as e:
    print(f"Error loading DeepLabV3+ model: {e}")
    model = None  # Handle the case where the model fails to load

# --- Preprocessing Transforms ---
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),  # Match the image size.  Crucial for consistency
    transforms.ToTensor(),                         # Convert to tensor and normalize to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize with ImageNet stats
])

def load_image(image_path):
    """Loads an image using PIL."""
    try:
        image = Image.open(image_path).convert("RGB")  # Convert to RGB to handle various formats
        return image
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def human_parsing(image_path):
    """
    Performs human parsing on an input image using a pre-trained DeepLabV3+ model.

    Args:
        image_path (str): Path to the input human image.

    Returns:
        torch.Tensor or None: A tensor representing the segmentation mask, or None if there was an error.
    """
    if model is None:
        print("Error: DeepLabV3+ model not loaded. Returning None.")
        return None

    try:
        input_image = load_image(image_path)
        if input_image is None:
            print(f"Error: Could not load image at {image_path}")
            return None

        input_tensor = preprocess(input_image)  # Apply preprocessing
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        if USE_CUDA and torch.cuda.is_available():
            input_batch = input_batch.cuda() # Move to GPU if available
            model.cuda()
        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0) # Get the predicted class for each pixel

        return output_predictions  # This is the segmentation mask

    except Exception as e:
        print(f"Error during human parsing: {e}")
        return None

def visualize_segmentation(image, mask, class_names,  alpha=0.5):
    """
    Visualizes the segmentation mask overlaid on the original image.

    Args:
        image (PIL.Image): The original image.
        mask (torch.Tensor): The segmentation mask (tensor of class indices).
        class_names (list): A list of class names corresponding to the mask indices.
        alpha (float): The transparency of the mask overlay (0 to 1).

    Returns:
        PIL.Image: The image with the segmentation mask overlaid.
    """
    if image is None or mask is None or len(class_names) == 0:
        print("Cannot visualize segmentation.  Image, mask, or class names missing.")
        return None

    try:
        # Convert image to numpy array if it's a PIL Image
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image  # Assuming it's already a numpy array

        # Create a color map for the segmentation classes
        num_classes = len(class_names)
        colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
        mask_np = mask.cpu().numpy()  # Move mask to CPU and convert to numpy
        masked_image = image_np.copy() # Create a copy to prevent from modifying original

        # Overlay the segmentation mask with colors
        for i in range(num_classes):
            color = colors[i]
            masked_image[mask_np == i] = (
                masked_image[mask_np == i] * (1 - alpha) + color * alpha
            ).astype(np.uint8)

        return Image.fromarray(masked_image) # Convert to PIL for display
    except Exception as e:
        print(f"Error visualizing segmentation: {e}")
        return None

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # Assuming you have a human image in your data/raw/human directory
    human_image_path = os.path.join(DATA_DIR, RAW_DATA_DIR, HUMAN_DIR, "human_001.png") # Replace with your image

    # Ensure the image exists
    if not os.path.exists(human_image_path):
        print(f"Error: Human image not found at {human_image_path}")
        sys.exit(1)  # Exit the script

    segmentation_mask = human_parsing(human_image_path)

    if segmentation_mask is not None:
        print("Segmentation Mask shape:", segmentation_mask.shape)

        # ---  Class Names (DeepLabV3+ uses the COCO dataset)---
        class_names = [
            '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor'
        ]

        # Visualize the segmentation
        from PIL import Image  # Import PIL locally
        original_image = load_image(human_image_path)
        if original_image:
            visualized_image = visualize_segmentation(original_image, segmentation_mask, class_names)

            if visualized_image:
                visualized_image.save("human_segmentation_output.png") # Save the output
                visualized_image.show() # Display the result
                print("Segmentation visualized and saved as human_segmentation_output.png")