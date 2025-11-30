import os
import cv2
import numpy as np


center_x = 40
y_line = 65  
pixel_length = 3.21

x_line = 70 
center_y = 40 

line_start = int(round(center_x - 5 * pixel_length))
line_end   = int(round(center_x + 5 * pixel_length))
line_start_h = int(round(center_x - 3.5 * pixel_length))  
line_end_h   = int(round(center_x + 3.5 * pixel_length)) 

y_line_start = int(round(center_y - 5 * pixel_length))
y_line_end   = int(round(center_y + 5 * pixel_length))
y_line_start_h = int(round(center_y - 3.5 * pixel_length))  
y_line_end_h   = int(round(center_y + 3.5 * pixel_length))


def get_image_sharpness(img):
    img = img.astype('uint8')
    factor = img.shape[0] * img.shape[1] * 1.0

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    normx = cv2.norm(sobelx)
    normy = cv2.norm(sobely)

    return (normx + normy) / factor


def draw_ruler_on_image(image):

    # No border added here

    # Horizontal line
    cv2.line(image, (line_start, y_line), (line_end, y_line), (0, 0, 0), 1)
    cv2.line(image, (line_start_h, y_line - 3), (line_start_h, y_line + 3), (0, 0, 0), 1)
    cv2.line(image, (line_end_h, y_line - 3), (line_end_h, y_line + 3), (0, 0, 0), 1)

    cv2.putText(image, "10", (line_end - 3, y_line + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)
    cv2.putText(image, "7", (line_end_h - 3, y_line + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)

    cv2.line(image, (line_start, y_line - 5), (line_start, y_line + 3), (0, 0, 0), 1)
    cv2.line(image, (line_end, y_line - 5), (line_end, y_line + 3), (0, 0, 0), 1)

    # Vertical line
    cv2.line(image, (x_line, y_line_start), (x_line, y_line_end), (0, 0, 0), 1)
    cv2.line(image, (x_line - 5, y_line_start), (x_line + 5, y_line_start), (0, 0, 0), 1)
    cv2.line(image, (x_line - 5, y_line_end), (x_line + 5, y_line_end), (0, 0, 0), 1)

    cv2.putText(image, "10", (x_line + 3, y_line_start + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)
    cv2.putText(image, "7", (x_line + 3, y_line_start_h + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)

    cv2.line(image, (x_line - 3, y_line_start_h), (x_line + 3, y_line_start_h), (0, 0, 0), 1)
    cv2.line(image, (x_line - 3, y_line_end_h), (x_line + 3, y_line_end_h), (0, 0, 0), 1)

    return image


def process_strip_with_ruler(input_strip, len_z_stack=9):
    print(input_strip.shape)
    h, w, c = input_strip.shape
    patch_w = w // len_z_stack

    patches = []
    sharpness_list = []

    # Extract patches & compute sharpness
    for i in range(len_z_stack):
        patch = input_strip[:, i * patch_w:(i + 1) * patch_w]
        sharp_val = get_image_sharpness(patch)
        sharpness_list.append(sharp_val)
        patches.append(patch)

    # Find sharpest patch
    best_idx = np.argmax(sharpness_list)

    # Draw ruler on the best patch
    patches[best_idx] = draw_ruler_on_image(patches[best_idx])

    # Add uniform bottom padding of 50px to ALL patches
    for i in range(len(patches)):
        patches[i] = cv2.copyMakeBorder(
            patches[i], top=0, bottom=50, left=0, right=0,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    # Concatenate safely
    final_strip = np.concatenate(patches, axis=1)
    return final_strip



# ---------------- Directory Processing ---------------- #

input_dir = "/Users/benjaminroyv/Downloads/data_something/data/c57654ee-8d89-4d4a-aa8f-5d173a69abea/ef5af20d-e922-4f1a-add1-343b71731368/wbc/Recon_data/data"
output_dir = "/Users/benjaminroyv/Downloads/data_something/data/c57654ee-8d89-4d4a-aa8f-5d173a69abea/ef5af20d-e922-4f1a-add1-343b71731368/wbc/Recon_data/data_with_strips"

os.makedirs(output_dir, exist_ok=True)

for folder_name in os.listdir(input_dir):
    if folder_name == ".DS_Store":
        continue

    folder_path = os.path.join(input_dir, folder_name)
    output_folder = os.path.join(output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    for image_file in os.listdir(folder_path):
        if image_file == ".DS_Store" or image_file.endswith(".pkl"):
            continue
        try:
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)

            final_strip = process_strip_with_ruler(image)

            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, final_strip)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue

print("✅ Processing completed successfully.")

