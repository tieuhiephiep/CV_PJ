import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import os

def load_and_preprocess_image(image_path, target_size=(48, 48)):
    """Load and preprocess a single image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return img

def load_dataset(data_dir, target_size=(48, 48), mode='train'):
    """
    Load and preprocess the dataset from directories.
    Args:
        data_dir: Base directory containing train and test folders
        target_size: Size to resize images to
        mode: 'train' or 'test'
    """
    X = []
    y = []
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    print(f"Loading {mode} images from directories...")
    for emotion_idx, emotion in enumerate(emotions):
        emotion_dir = os.path.join(data_dir, mode, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Warning: Directory {emotion_dir} does not exist")
            continue
            
        print(f"Loading {emotion} images...")
        for img_name in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_name)
            img = load_and_preprocess_image(img_path, target_size)
            if img is not None:
                X.append(img)
                y.append(emotion_idx)
    
    if len(X) == 0:
        raise ValueError(f"No {mode} images were loaded. Please check your data directory structure.")
        
    X = np.array(X)
    y = to_categorical(y, num_classes=len(emotions))
    print(f"Loaded {len(X)} {mode} images")
    return X, y

def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.show()

def draw_emotion(frame, face_rect, emotion, confidence):
    """Draw emotion label and confidence on frame."""
    x, y, w, h = face_rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    label = f"{emotion}: {confidence:.2f}"
    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def get_emotion_color(emotion):
    """Get color for different emotions."""
    colors = {
        'angry': (0, 0, 255),    # Red
        'disgust': (0, 128, 0),  # Green
        'fear': (128, 0, 128),   # Purple
        'happy': (0, 255, 255),  # Yellow
        'sad': (255, 0, 0),      # Blue
        'surprise': (255, 165, 0),# Orange
        'neutral': (255, 255, 255)# White
    }
    return colors.get(emotion, (255, 255, 255)) 