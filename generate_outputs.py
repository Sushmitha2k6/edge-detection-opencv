import os
import shutil
import cv2
import numpy as np

os.environ['MPLCONFIGDIR'] = r'C:\Users\LENOVO\.gemini\antigravity\brain\b768dafa-3595-483f-97f3-a6502a8b1ef6\scratch'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo_dir = r'C:\Users\LENOVO\.gemini\antigravity\scratch\edge-detection-opencv'
assets_dir = os.path.join(repo_dir, 'assets')
os.makedirs(assets_dir, exist_ok=True)

src_img = r'C:\Users\LENOVO\.gemini\antigravity\brain\b768dafa-3595-483f-97f3-a6502a8b1ef6\.user_uploaded\media_1795587128684_89aa24d1.png'
dest_img = os.path.join(repo_dir, 'nature.jpeg')

shutil.copyfile(src_img, dest_img)
shutil.copyfile(src_img, os.path.join(assets_dir, 'nature.jpeg'))

# Read image
image = cv2.imread(dest_img)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Original Image
fig, ax = plt.subplots(figsize=(5, 7.5))
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax.set_title('Original Image')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'output_1_original.png'), bbox_inches='tight', dpi=150)
plt.close()

# 2. Sobel Edge Detector
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

fig, ax = plt.subplots(figsize=(5, 7.5))
ax.imshow(sobel_combined, cmap='gray')
ax.set_title('Sobel Edge Detection')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'output_2_sobel.png'), bbox_inches='tight', dpi=150)
plt.close()

# 3. Prewitt Edge Detector
prewitt_x = np.array([[-1, 0, -1],
                      [ 1, 0, -1],
                      [ 1, 0, -1]])

prewitt_y = np.array([[ 1,  1,  1],
                      [ 0,  0,  0],
                      [-1, -1, -1]])

prewitt_x_edge = cv2.filter2D(gray_image, -1, prewitt_x)
prewitt_y_edge = cv2.filter2D(gray_image, -1, prewitt_y)

prewitt = cv2.magnitude(prewitt_x_edge.astype(np.float32),
                        prewitt_y_edge.astype(np.float32))

fig, ax = plt.subplots(figsize=(5, 7.5))
ax.imshow(prewitt, cmap='gray')
ax.set_title('Prewitt Edge Detection')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'output_3_prewitt.png'), bbox_inches='tight', dpi=150)
plt.close()

# 4. Roberts Edge Detector
roberts_x = np.array([[1, 0],
                      [0, -1]])

roberts_y = np.array([[0, 1],
                      [-1, 0]])

roberts_x_edge = cv2.filter2D(gray_image, -1, roberts_x)
roberts_y_edge = cv2.filter2D(gray_image, -1, roberts_y)

roberts = cv2.magnitude(
    roberts_x_edge.astype(np.float32),
    roberts_y_edge.astype(np.float32)
)

fig, ax = plt.subplots(figsize=(5, 7.5))
ax.imshow(roberts, cmap='gray')
ax.set_title('Roberts Edge Detection')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'output_4_roberts.png'), bbox_inches='tight', dpi=150)
plt.close()

# 5. Laplacian Edge Detector
Laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

fig, ax = plt.subplots(figsize=(5, 7.5))
ax.imshow(Laplacian, cmap='gray')
ax.set_title('Laplacian Edge Detection')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'output_5_laplacian.png'), bbox_inches='tight', dpi=150)
plt.close()

# 6. Canny Edge Detector
Canny_edges = cv2.Canny(gray_image, 50, 150)

fig, ax = plt.subplots(figsize=(5, 7.5))
ax.imshow(Canny_edges, cmap='gray')
ax.set_title('Canny Edge Detection')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'output_6_canny.png'), bbox_inches='tight', dpi=150)
plt.close()

print('All edge detection outputs generated successfully!')
