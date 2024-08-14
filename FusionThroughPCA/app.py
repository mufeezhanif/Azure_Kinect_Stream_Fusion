
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def resize_image(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def enhance_image(rgb, ir):
    ir_resized = resize_image(ir, (rgb.shape[1], rgb.shape[0]))

    rgb_normalized = normalize_image(rgb)
    ir_normalized = normalize_image(ir_resized)
    
    rgb_flat = rgb_normalized.reshape(-1, 3)
    ir_flat = ir_normalized.flatten().reshape(-1, 1)

    combined_features = np.hstack((rgb_flat, ir_flat))

    pca = PCA(n_components=3)
    fused_features = pca.fit_transform(combined_features)

    fused_image = fused_features[:, :3].reshape(rgb.shape[0], rgb.shape[1], 3)

    fused_image_normalized = normalize_image(fused_image)

    return fused_image_normalized

rgb_image = cv2.imread('assets/rgb/FLIR_00002.jpg')
ir_image = cv2.imread('assets/ir/FLIR_00002.jpeg', cv2.IMREAD_GRAYSCALE)

enhanced_rgb_image = enhance_image(rgb_image, ir_image)

# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original RGB Image")
plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title("IR Image")
plt.imshow(ir_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Enhanced RGB Image")
imagetobeDisplayed =cv2.cvtColor((enhanced_rgb_image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
cv2.imwrite('output.jpg',imagetobeDisplayed) 
plt.imshow(cv2.cvtColor((enhanced_rgb_image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

print(f'rgb rgb shape {rgb_image.shape}')
print(f'rgb depth shape {ir_image.shape}')
print(f'rgb ir shape {imagetobeDisplayed.shape}')
plt.show()