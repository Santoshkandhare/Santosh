FILTERS:
# Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Loading Image
image = cv2.imread(r".\\sunset.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
x, y = gray.shape[:2]


Creating and Applying Low Pass Average Filter
kernel = np.ones([3, 3], dtype = int)
kernel = kernel / 9

out1 = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

out2 = cv2.blur(src=image, ksize=(3,3))  # 5x5, 7x7, 11x11

plt.imshow(image, 'gray')
plt.title('Input Image')
plt.show()




plt.imshow(out1, 'gray')
plt.title('Low Pass Average Filter')
plt.show()

plt.imshow(out2, 'gray')
plt.title('Low Pass Average Filter - In built function')
plt.show()


# Low Pass Median Filter

out = cv2.medianBlur(src=image, ksize = 3)  # 5, 7, 9, 11

plt.imshow(image, 'gray')
plt.title('Input Image')
plt.show()

plt.imshow(out, 'gray')
plt.title('Low Pass Median Filter')
plt.show()

# High Pass Filter
kernel = np.array([[-1/9, -1/9, -1/9],
                   [-1/9, 8/9, -1/9],
                   [-1/9, -1/9, -1/9]])

# another kernal without 1/9
out = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

plt.imshow(image, 'gray')
plt.title('Input Image')
plt.show()

plt.imshow(out, 'gray')
plt.title('High Pass Filter')
plt.show()

# High Boost Filter
kernel = np.ones([3, 3], dtype = int)
kernel = kernel / 9

blur_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

out = cv2.addWeighted(image, 2, blur_image, -1, 0) #


plt.imshow(image, 'gray')
plt.title('Input Image')
plt.show()

plt.imshow(out, 'gray')
plt.title('High Boost Filter')
plt.show()


# NOISE:
# Adding Random Noise
# Read Image
img = cv2.imread('lily.jpg') # Color image

# Convert the image to grayscale
img_gray = img[:,:,1]

plt.imshow(img_gray,'gray')
plt.title('Original Image')
plt.show()

# Genearte noise with same shape as that of the image
noise = np.random.normal(0, 50, img_gray.shape)

# Add the noise to the image
img_noised = img_gray + noise

# Clip the pixel values to be between 0 and 255.
img_noised = np.clip(img_noised, 0, 255).astype(np.uint8)

plt.imshow(img_noised,'gray')
plt.title('Noise Added Image')
plt.show()



#Salt and Paper Noise
# Read Image
img = cv2.imread('lily.jpg') # Color image

# Convert the image to grayscale
img_gray = img[:,:,1]

plt.imshow(img_gray,'gray')
plt.title('Original Image')
plt.show()

# Get the image size (number of pixels in the image).
img_size = img_gray.size

# Set the percentage of pixels that should contain noise
noise_percentage = 0.1  # Setting to 10%

# Determine the size of the noise based on the noise precentage
noise_size = int(noise_percentage*img_size)

# Randomly select indices for adding noise.
random_indices = np.random.choice(img_size, noise_size)

# Create a copy of the original image that serves as a template for the noised image.
img_noised = img_gray.copy()

# Create a noise list with random placements of min and max values of the image pixels.
noise = np.random.choice([img_gray.min(), img_gray.max()], noise_size)

# Replace the values of the templated noised image at random indices with the noise, to obtain the final noised image.
img_noised.flat[random_indices] = noise

plt.imshow(img_noised,'gray')
plt.title('Noise Added Image')
plt.show()

# Removing Noise with Median and Mean LPF:
img = cv2.imread("C:\\Users\\kambl\\Computer_Vision_OpenCV\\Sample Images\\lenna.png", cv2.IMREAD_GRAYSCALE)

# Add Salt and Pepper Noise
noise_percentage = 0.1  # 10% noise
noise_size = int(noise_percentage * img.size)

# Randomly select pixel indices for noise
random_indices = np.random.choice(img.size, noise_size, replace=False)
img_noised = img.copy()
img_noised.flat[random_indices] = np.random.choice([0, 255], noise_size)

# Display the noised image
plt.imshow(img_noised, cmap='gray')
plt.title('Noised Image (Salt & Pepper)')
plt.show()

# Remove noise using Median Filter
img_median = cv2.medianBlur(img_noised, 5)

# Display the image after Median filtering
plt.imshow(img_median, cmap='gray')
plt.title('Denoised Image (Median Filter)')
plt.show()

# Remove noise using Mean Filter
img_mean = cv2.blur(img_noised, (5, 5))

# Display the image after Mean filtering
plt.imshow(img_mean, cmap='gray')
plt.title('Denoised Image (Mean Filter)')
plt.show()



# Gaussian Noise
# Load the image
image = cv2.imread('.\lily.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
x,y = gray.shape[:2]

plt.imshow(gray,'gray')
plt.title('Original Image')
plt.show()

# Generate random Gaussian noise
mean = 0
stddev = 180
noise = np.zeros(img.shape, np.uint8)
cv2.randn(noise, mean, stddev)

# Add noise to image
noisy_img = cv2.add(img, noise)

# Save noisy image

plt.imshow(noisy_img,'gray')
plt.title('Noise Added Image')
plt.show()

# Remove Noise using Gaussian Blur
image = cv2.imread("C:\\Users\\kambl\\Computer_Vision_OpenCV\\Sample Images\\lenna.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.show()

# Generate random
mean = 0
stddev = 30  
noise = np.zeros_like(gray, np.uint8)
cv2.randn(noise, mean, stddev)

# Add noise to the image
noisy_img = cv2.add(gray, noise)

plt.imshow(noisy_img, cmap='gray')
plt.title('Noise Added Image')
plt.show()

denoised_img = cv2.GaussianBlur(noisy_img, (5, 5), 0)

plt.imshow(denoised_img, cmap='gray')
plt.title('Denoised Image using Gaussian Blur')
plt.show()

#Add impulse Noise and use Filter to remove it
# Load the image
image = cv2.imread('.\lily.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
x,y = gray.shape[:2]

plt.imshow(gray,'gray')
plt.title('Original Image')
plt.show()

imp_noise=np.zeros((x,y),dtype=np.uint8)
cv2.randu(imp_noise,0,255)
imp_noise=cv2.threshold(imp_noise,245,255,cv2.THRESH_BINARY)[1]

in_img=cv2.add(gray,imp_noise)

fig=plt.figure(dpi=200)

fig.add_subplot(1,3,1)
plt.imshow(img,cmap='gray')
plt.axis("off")
plt.title("Original")

fig.add_subplot(1,3,2)
plt.imshow(imp_noise,cmap='gray')
plt.axis("off")
plt.title("Impulse Noise")

fig.add_subplot(1,3,3)
plt.imshow(in_img,cmap='gray')
plt.axis("off")
plt.title("Combined")


