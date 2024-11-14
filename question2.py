import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern
from scipy.signal import stft
from sklearn.decomposition import PCA
from os import listdir
from os.path import isfile, join
from skimage import color

# Load image files from a directory
image_folder = 'path_to_your_image_folder'  # Update with your image folder path
image_files = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]

# Initialize list for features and labels
features = []
labels = []  # Assuming the label is encoded in the filename (e.g., cat_1.jpg, dog_2.jpg)

# Feature extraction functions
def extract_hog(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute HOG features
    features, _ = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def extract_stft(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply STFT on each row (or column)
    f, t, Zxx = stft(gray_image.flatten(), fs=1.0, nperseg=256)
    # Flatten the spectrogram for feature vector
    return np.abs(Zxx.flatten())

def extract_lbp(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute LBP features
    radius = 3
    n_points = 24
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    # Compute the histogram of LBP
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    # Normalize the histogram
    return lbp_hist / np.sum(lbp_hist)

# Loop over the image files and extract features
for img_file in image_files:
    # Read the image
    img_path = join(image_folder, img_file)
    img = cv2.imread(img_path)

    # Resize image to a consistent size
    img = cv2.resize(img, (128, 128))

    # Extract features
    hog_features = extract_hog(img)
    stft_features = extract_stft(img)
    lbp_features = extract_lbp(img)

    # Concatenate features from all methods
    combined_features = np.hstack((hog_features, stft_features, lbp_features))

    # Assuming the label is in the image filename (you can adjust this based on your dataset)
    label = img_file.split('_')[0]  # Adjust this according to your file naming convention

    # Append features and label
    features.append(combined_features)
    labels.append(label)

# Convert features to a numpy array
features = np.array(features)

# Apply PCA for dimensionality reduction (e.g., reduce to 50 components)
pca = PCA(n_components=50)
reduced_features = pca.fit_transform(features)

# Create DataFrame with reduced features
df = pd.DataFrame(reduced_features)
df['label'] = labels  # Add label to the DataFrame

# Save the features and labels to CSV
df.to_csv('image_features.csv', index=False)

print("Feature extraction and CSV generation complete!")
