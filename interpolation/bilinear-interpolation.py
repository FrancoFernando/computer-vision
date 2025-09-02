import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_small_colorful_grid_for_interpolation(grid_size=9):
    """
    Create a small colorful grid specifically for interpolation examples
    """
    image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    
    colors = [
        [255, 87, 34], [103, 58, 183], [96, 125, 139], [121, 85, 72],
        [0, 188, 212], [233, 30, 99], [76, 175, 80], [33, 150, 243],
        [255, 235, 59], [156, 39, 176], [244, 67, 54], [139, 195, 74],
        [255, 152, 0], [63, 81, 181], [0, 150, 136], [158, 158, 158],
        [255, 193, 7], [3, 169, 244], [205, 220, 57], [255, 64, 129],
    ]
    
    # Each pixel gets a random color 
    for i in range(grid_size):
        for j in range(grid_size):
            color = colors[np.random.randint(0, len(colors))]
            image[i, j] = color
    
    return image

def create_colorful_grid(grid_size=10, cell_size=10):
    """
    Create a colorful grid image with random colors in each cell
    and small black dots at grid intersections
    """
    image_size = grid_size * cell_size
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    
    # Define a palette of vibrant colors (RGB format)
    colors = [
        [255, 87, 34],   # Orange
        [103, 58, 183],  # Purple
        [96, 125, 139],  # Blue Grey
        [121, 85, 72],   # Brown
        [0, 188, 212],   # Cyan
        [233, 30, 99],   # Pink
        [76, 175, 80],   # Green
        [33, 150, 243],  # Blue
        [255, 235, 59],  # Yellow
        [156, 39, 176],  # Purple
        [244, 67, 54],   # Red
        [139, 195, 74],  # Light Green
        [255, 152, 0],   # Orange
        [63, 81, 181],   # Indigo
        [0, 150, 136],   # Teal
        [158, 158, 158], # Grey
        [255, 193, 7],   # Amber
        [3, 169, 244],   # Light Blue
        [205, 220, 57],  # Lime
        [255, 64, 129],  # Pink
    ]
    
    # Fill each cell with a random color
    for i in range(grid_size):
        for j in range(grid_size):
            # Choose a random color from the palette
            color = colors[np.random.randint(0, len(colors))]
            
            # Define cell boundaries
            y_start = i * cell_size
            y_end = (i + 1) * cell_size
            x_start = j * cell_size
            x_end = (j + 1) * cell_size
            
            # Fill the cell with the chosen color
            image[y_start:y_end, x_start:x_end] = color
    
    # Add small black dots at grid intersections
    # dot_size = 3
    # for i in range(grid_size + 1):
    #     for j in range(grid_size + 1):
    #         y = i * cell_size
    #         x = j * cell_size
    #         if y < image_size and x < image_size:
    #             # Draw a small black dot
    #             y_start = max(0, y - dot_size // 2)
    #             y_end = min(image_size, y + dot_size // 2 + 1)
    #             x_start = max(0, x - dot_size // 2)
    #             x_end = min(image_size, x + dot_size // 2 + 1)
    #             image[y_start:y_end, x_start:x_end] = [0, 0, 0]  # Black
    
    return image

def demonstrate_bilinear_interpolation():
    # Load an image
    image = cv2.imread('input_image.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get original dimensions
    height, width = image_rgb.shape[:2]
    print(f"Original image size: {width}x{height}")
    
    # Resize image using bilinear interpolation (cv2.INTER_LINEAR)
    new_width = 800
    new_height = 600
    resized_bilinear = cv2.resize(image_rgb, (new_width, new_height), 
                                 interpolation=cv2.INTER_LINEAR)
    
    # For comparison, also resize using nearest neighbor
    resized_nearest = cv2.resize(image_rgb, (new_width, new_height), 
                                interpolation=cv2.INTER_NEAREST)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(resized_bilinear)
    axes[1].set_title('Bilinear Interpolation')
    axes[1].axis('off')
    
    axes[2].imshow(resized_nearest)
    axes[2].set_title('Nearest Neighbor (for comparison)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return resized_bilinear

# Manual implementation to show what happens under the hood
def manual_bilinear_resize(image, new_width, new_height):
    """
    Manual implementation of bilinear interpolation for educational purposes
    """
    height, width, channels = image.shape
    resized = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    
    # Calculate scaling factors
    x_scale = width / new_width
    y_scale = height / new_height
    
    for new_y in range(new_height):
        for new_x in range(new_width):
            # Map new coordinates to original image coordinates
            orig_x = new_x * x_scale
            orig_y = new_y * y_scale
            
            # Get integer parts (corners of the cell)
            x0 = int(orig_x)
            y0 = int(orig_y)
            x1 = min(x0 + 1, width - 1)
            y1 = min(y0 + 1, height - 1)
            
            # Get fractional parts (tx, ty)
            tx = orig_x - x0
            ty = orig_y - y0
            
            # Get the four corner pixels
            c00 = image[y0, x0]
            c10 = image[y0, x1]
            c01 = image[y1, x0]
            c11 = image[y1, x1]
            
            # Perform bilinear interpolation
            # First, interpolate along x-axis
            a = c00 * (1 - tx) + c10 * tx
            b = c01 * (1 - tx) + c11 * tx
            
            # Then interpolate along y-axis
            result = a * (1 - ty) + b * ty
            
            resized[new_y, new_x] = result.astype(np.uint8)
    
    return resized

# Example usage
if __name__ == "__main__":
    # Create a simple test image with a gradient
    # test_image = np.zeros((10, 10, 3), dtype=np.uint8)
    # for i in range(10):
    #     for j in range(10):
    #         test_image[i, j] = [i * 2.55, j * 2.55, 128]
    
    # Resize using OpenCV
    interpolated_image_size = 10

    test_image = create_colorful_grid(10, 50)
    opencv_result = cv2.resize(test_image, (interpolated_image_size, interpolated_image_size), interpolation=cv2.INTER_LINEAR)

    # Resize using our manual implementation
    manual_result = manual_bilinear_resize(test_image, interpolated_image_size, interpolated_image_size)

    # Display comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(test_image)
    axes[0].set_title(f'Original ({test_image.shape[1]}x{test_image.shape[0]})')
    axes[0].axis('off')
    
    axes[1].imshow(opencv_result)
    axes[1].set_title(f'OpenCV Bilinear ({opencv_result.shape[1]}x{opencv_result.shape[0]})')
    axes[1].axis('off')
    
    axes[2].imshow(manual_result)
    axes[2].set_title(f'Manual Bilinear ({manual_result.shape[1]}x{manual_result.shape[0]})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

