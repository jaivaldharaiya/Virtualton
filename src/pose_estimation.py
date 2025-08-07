# src/pose_estimation.py
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from config import *  # Import configuration

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,  # Set to True if processing a single image
                     model_complexity=2,    # Adjust for accuracy/speed (0, 1, 2)
                     smooth_landmarks=True,
                     min_detection_confidence=0.5, # Confidence thresholds
                     min_tracking_confidence=0.5)

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
def mediapipe_pose_estimation(image_path):
    """
    Estimates pose keypoints using MediaPipe BlazePose.

    Args:
        image_path (str): Path to the input image.

    Returns:
        list or None: A list of keypoint coordinates (x, y) and confidence scores,
                      or None if an error occurred. The keypoint order is defined by MediaPipe.
    """
    try:
        image = load_image(image_path) # Load the image using PIL
        if image is None:
            print(f"Error: Could not load image at {image_path}")
            return None

        # Convert the PIL image to a NumPy array (OpenCV format: BGR)
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to BGR

        # Process the image with MediaPipe Pose
        results = pose.process(image_np)

        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                # MediaPipe returns normalized coordinates (0.0 to 1.0)
                # Scale to image dimensions
                height, width, _ = image_np.shape
                x = landmark.x * width
                y = landmark.y * height
                keypoints.append((x, y))  # Append keypoint
            return keypoints
        else:
            print("No pose detected.")
            return None

    except Exception as e:
        print(f"Error during pose estimation: {e}")
        return None

def visualize_keypoints(image, keypoints, output_path="pose_output.png", radius=5, color=(0, 255, 0), thickness=2):
    """
    Visualizes the keypoints on the image and saves the result.

    Args:
        image (PIL.Image or numpy.ndarray): The original image.
        keypoints (list): A list of (x, y) keypoint coordinates.
        output_path (str): The path to save the output image.
        radius (int): The radius of the circles for keypoints.
        color (tuple): The color of the keypoint circles (BGR).
        thickness (int): The thickness of the circle outlines.
    """
    try:
        # Convert the image to a NumPy array if it's a PIL Image
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_np = image  # Assuming it's already a NumPy array

        if keypoints:
            for x, y in keypoints:
                cv2.circle(image_np, (int(x), int(y)), radius, color, thickness)
        # Convert back to RGB for saving with PIL (optional)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil.save(output_path)

        print(f"Keypoints visualized and saved to {output_path}")

    except Exception as e:
        print(f"Error visualizing keypoints: {e}")
        return None


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    human_image_path = os.path.join(DATA_DIR, RAW_DATA_DIR, HUMAN_DIR, "human_001.png") # Replace with your image
    if not os.path.exists(human_image_path):
        print(f"Error: Human image not found at {human_image_path}")
        sys.exit(1)

    keypoints = mediapipe_pose_estimation(human_image_path)

    if keypoints:
        print(f"Detected {len(keypoints)} keypoints.")
        # Print a few keypoint examples to verify
        for i in range(min(5, len(keypoints))):
            print(f"Keypoint {i+1}: {keypoints[i]}")

        visualize_keypoints(load_image(human_image_path), keypoints, output_path="keypoints_output.png")