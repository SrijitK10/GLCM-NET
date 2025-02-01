import numpy as np
def calculate_glm(image, distance_x=1, distance_y=1, levels=256):
    """
    Calculates the Gray-Level Co-Occurrence Matrix (GLCM) for a grayscale image.
    Parameters:
    - image (2D array): Input grayscale image.
    - distance_x (int): Horizontal distance between the reference pixel and its neighbor.
    - distance_y (int): Vertical distance between the reference pixel and its neighbor.
    - levels (int): Number of intensity levels (usually 256 for 8-bit grayscale images).
    Returns:
    - gIcm (2D array): Gray-Level Co-Occurrence Matrix (GLCM).
    """
# Initialize the GLCM matrix
    glcm = np.zeros((levels, levels), dtype='int64')
# Prepare images for matricial operations
    sx, sy = image.shape
    image1_ready = image[0:sx - distance_x, 0:sy - distance_y]
    image2_ready = image[distance_x:, distance_y:]
# Populate the GLCM 

    for i in range(levels):
        image2_ready_temp = image2_ready[image1_ready == i]
        for j in range(levels):
            glcm[i, j] = np.sum(image2_ready_temp == j)
    

    return glcm