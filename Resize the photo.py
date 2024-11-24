import numpy as np
import cv2

image_path = (r"D:\University\Machine Vision Repositories\imageextractor\forest_picture.jpg")
original_image = cv2.imread(image_path)

resize_dimensions = (900, 900)
resized_image = cv2.resize(original_image, resize_dimensions, interpolation=cv2.INTER_LINEAR)

def convert_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_image = convert_to_grayscale(resized_image)

original_width, original_height = resize_dimensions

small_width, small_height = 300, 300

small_images = [np.zeros((small_width, small_height), np.uint8) for _ in range(9)]

def get_neighborhood(data, y, x):
    return data[y-1:y+2, x-1:x+2]

for y in range(1, original_height, 3):
    for x in range(1, original_width, 3):
        neighborhood = get_neighborhood(gray_image, y, x)
        col, row = (x // 3) % small_width, (y // 3) % small_height
        small_images[0][row, col] = neighborhood[0, 0]
        small_images[1][row, col] = neighborhood[0, 1]
        small_images[2][row, col] = neighborhood[0, 2]
        small_images[3][row, col] = neighborhood[1, 0]
        small_images[4][row, col] = neighborhood[1, 1]
        small_images[5][row, col] = neighborhood[1, 2]
        small_images[6][row, col] = neighborhood[2, 0]
        small_images[7][row, col] = neighborhood[2, 1]
        small_images[8][row, col] = neighborhood[2, 2]

cv2.imshow("Original Image", gray_image)
for i, img in enumerate(small_images, 1):
    cv2.imshow(f"Small Image {i}", img)
cv2.waitKey(0)
