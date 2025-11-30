import cv2,os
import numpy as np
import pickle

def get_image_sharpness(img):
    img = img.astype('uint8')
    factor = img.shape[0] * img.shape[1] * 1.0
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    normx = cv2.norm(sobelx)
    normy = cv2.norm(sobely)
    sharpness_overall = (abs(normx) + abs(normy)) / factor
    return sharpness_overall

def extract_input_patch(image, len_z_stack = 9):
    h, w, c = image.shape

    patch_w = w // len_z_stack
    patch_h = patch_w
    img_count_sharpness_pair = []

    for i in range(0,(len_z_stack)):
        img = image[0 : patch_h, i * patch_w : patch_w + i * patch_w]
        sharpness = get_image_sharpness(img)
        img_count_sharpness_pair.append((i, sharpness))

    # Sort the list of tuples based on the second element (float variable)
    sorted_data = sorted(img_count_sharpness_pair, key=lambda x: x[1])
    max_index = len(sorted_data) - 1
    max_count = sorted_data[max_index][0]
    input_patch = image[0 : patch_h, max_count * patch_w : patch_w + max_count * patch_w]
 
    return input_patch


input_dir = "/Volumes/Sigvet_DC/pd10/BO169/01_09_25/697b9f71-b3ff-41da-9536-c359a20d13e1"

output_dir = "/Volumes/Sigvet_DC/pd10/BO169/patches"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



# for image_files in os.listdir(input_dir):

    
  
        # print(image_files)
        #if image_files == "strips":
        # if image_files == ".DS_Store":
        #     continue
        # path = os.path.join(input_dir,image_files)
        # merge_target = os.path.join(output_dir, image_files)
        # if not os.path.exists(merge_target):
        #     os.makedirs(merge_target)              
for image_file in os.listdir(input_dir):
    try:
        if image_file == ".DS_Store":
            continue
        print(image_file)
        image_path = os.path.join(input_dir,image_file)
        image = cv2.imread(image_path)

        # with open(image_path, 'rb') as f:
        #     image = pickle.load(f)
        #     print(image)
        patch = extract_input_patch(image)
        cv2.imwrite(os.path.join(output_dir,image_file), patch)

    
    except Exception as e:
        print(e)
        continue

print('Processing completed.')