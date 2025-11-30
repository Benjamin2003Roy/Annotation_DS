import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from WBC_sigvet_inference.WBC_sigvet_inference.classification import classify

def mount_smb_share(share_name, mount_point):
    server = "192.168.0.253"
    username = "Sigtuple"
    password = "Sigtuple@123"

    if not os.path.ismount(mount_point):
        print(f"Mounting //{server}/{share_name} at {mount_point} ...")
        os.makedirs(mount_point, exist_ok=True)
        smb_url = f"//{server}/{share_name}"
        cmd = [
            "sudo", "mount", "-t", "cifs", smb_url, mount_point,
            "-o", f"username={username},password={password},rw"
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Mounted {share_name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to mount {share_name}: {e}")
            raise
    else:
        print(f"✅ Already mounted: {mount_point}")
    return mount_point
#     "d98b501b-5ec5-45f9-b793-14ffbc880363",
    # "dc3a1be9-9d37-43ca-b203-11209642fd59",
    # "6962b23a-70ee-48c8-b122-a8733c4f3c79",
    # "ffe59e45-71a8-4b9a-b389-f7b4c75c04f7",
    # "42b3aca3-863b-43e2-8d21-20428db2cff3",
    # "26f541a2-7a02-4df3-87a9-83dcd78f2e06",
    # "46edbf1b-2353-40d2-9656-c014e3a0e958",
    # "6ca69315-c47e-4065-9132-5cae40eb9bc5",
    # "9b671e98-62b4-471b-abab-4ca2d58acb22",
    # "4281ee0d-4ef8-4584-8724-2fce3341418c",
    # "6a853a61-3f89-41e1-91b5-9c951d744533",

# File names and accession numbers
file_names = [
    "2a0bdff8-060a-4de6-98fb-05a917fb47db",
]

accession_numbers = [
    "HE1266"
]

# file_names =  ["206823f3-ec76-4ff8-a5b9-a1717cc72cf6","9cb518c3-124d-45cd-b9f2-09eefaf15f3b","df1b4ac2-058f-46c8-a6a2-ad53114a76a7","da3fea6d-8e46-42ee-8dbb-2f87fa3787d6","7071cc78-d017-4f60-b912-1e3e946af128","34bc2f57-bf46-45b0-b5a1-4f11532f2d62","cfbf029a-854c-46a3-9ffd-38765fd03ce0","2e1c13df-ca4d-4ee6-918a-3f1a446fcb59","eb58f9e4-a915-404e-8fbd-596a5e1bf5d1","e145bff2-a7dd-442f-9a09-c1c4ba06c1e6","a06b2a20-5a98-4a3b-acaa-0fa46b610a3a","0230e142-d735-4b1b-b7ce-466e820180d0","8c0a4eb6-6d3c-4ddc-a165-ee428060114a","5a1d6d8e-5230-4d9b-b68b-4405ff1bc726","57890956-8af8-4a83-9938-07f5a016a9ff","a08191c3-db97-4c2d-b9ef-d68ad37a17dc","7d474ebe-6720-4b23-8a92-68c012fb42ba","3461a523-97fd-4c0d-9dbf-2582d11ed177","f2a99fdd-7921-4163-8951-c3ab5a456ee3","a9e5c8fe-2245-45be-9372-ee2911f683f6"
# ]

# accession_numbers = [
#     "HE1291",
#     "HE1293",
#     "HE1281",
#     "PA872",
#     "PA873",
#     "HE1283",
#     "PA876",
#     "HE1298",
#     "HE1313",
#     "HE1300",
#     "HE1301",
#     "CA431",
#     "HE1325",
#     "PA894",
#     "HE1258",
#     "HE1271",
#     "HE1297",
#     "PA897",
#     "PA892",
#     "PA893"
# ]



# Helper functions
def get_image_sharpness(img):
    img = img.astype('uint8')
    factor = img.shape[0] * img.shape[1] * 1.0
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    normx = cv2.norm(sobelx)
    normy = cv2.norm(sobely)
    sharpness_overall = (abs(normx) + abs(normy)) / factor
    return sharpness_overall


def extract_input_patch(image, len_z_stack=9):
    h, w, c = image.shape
    patch_w = w // len_z_stack
    patch_h = patch_w
    img_count_sharpness_pair = []

    for i in range(len_z_stack):
        img = image[0:patch_h, i * patch_w:patch_w + i * patch_w]
        sharpness = get_image_sharpness(img)
        img_count_sharpness_pair.append((i, sharpness))

    sorted_data = sorted(img_count_sharpness_pair, key=lambda x: x[1])
    max_count = sorted_data[-1][0]
    input_patch = image[0:patch_h, max_count * patch_w:patch_w + max_count * patch_w]
    return input_patch


def patch_extraction(input_dir):
    # Input: /path/accession_number/without_scale
    # Output: /path/accession_number/patches
    parent_dir = os.path.dirname(input_dir)
    output_dir = os.path.join(parent_dir, "patches")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return None

    for image_file in os.listdir(input_dir):
        if image_file == ".DS_Store":
            continue
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        patch = extract_input_patch(image)
        cv2.imwrite(os.path.join(output_dir, image_file), patch)

    print(f"[INFO] Patch extraction completed: {output_dir}")
    return output_dir


def scaled_strips_in_classes_folder(output_path_classify, des_path_with_scale):
    # Input: /path/accession_number/classify
    # Output: /path/accession_number/scaled_strips_in_classes_folder
    parent_dir = os.path.dirname(output_path_classify)
    output_path = os.path.join(parent_dir, "scaled_strips_in_classes_folder")
    os.makedirs(output_path, exist_ok=True)
    
    if not os.path.exists(output_path_classify) or not os.path.exists(des_path_with_scale):
        print(f"Required directories do not exist")
        return
    
    # Get list of files in with_scale directory once
    with_scale_files = set(os.listdir(des_path_with_scale))
    
    for classes in os.listdir(output_path_classify):
        if classes == ".DS_Store":
            continue
        classes_folder_path = os.path.join(output_path_classify, classes)
        if not os.path.isdir(classes_folder_path):
            continue
            
        for file_name in os.listdir(classes_folder_path):
            if file_name == ".DS_Store":
                continue
            if file_name in with_scale_files:
                class_path_in_scaled_strip = os.path.join(output_path, classes)
                file_source_path = os.path.join(des_path_with_scale, file_name)
                
                os.makedirs(class_path_in_scaled_strip, exist_ok=True)
                shutil.copy(file_source_path, class_path_in_scaled_strip)
    
    print(f"[INFO] Scaled strips in classes folder completed: {output_path}")


def process_accession_without_scale(idx, nats_path, base_dir):
    file_name = file_names[idx]
    accession_number = accession_numbers[idx]
    
    # Create output directory: base_dir/accession_number/without_scale
    output_dir = os.path.join(base_dir, accession_number, "without_scale")
    os.makedirs(output_dir, exist_ok=True)

    found_data = False
    
    for x in os.listdir(nats_path):
        if not os.path.isdir(os.path.join(nats_path, x)) or x == ".DS_Store":
            continue
        if x != accession_number:
            continue

        for date in os.listdir(os.path.join(nats_path, x)):
            date_path = os.path.join(nats_path, x, date)
            if not os.path.isdir(date_path):
                continue
                
            for tray_id in os.listdir(date_path):
                tray_path = os.path.join(date_path, tray_id)
                if not os.path.isdir(tray_path):
                    continue                  
                for y in os.listdir(tray_path):
                    if y != file_name:
                        continue

                    data_path = os.path.join(tray_path, y)
                    pkl_stack_path = os.path.join(data_path, "wbc", "Recon_data", "pkl_stacks")
                    
                    if not os.path.exists(pkl_stack_path):
                        print(pkl_stack_path)
                        print("[INFO] Path Not Exist")
                        continue

                    pkl_files = [f for f in os.listdir(pkl_stack_path) if f.lower().endswith(".pkl")]
                    for pkl_file in pkl_files:
                        pkl_path = os.path.join(pkl_stack_path, pkl_file)
                        try:
                            with open(pkl_path, 'rb') as f:
                                data = pickle.load(f)

                            for stack_number in data['patch_stacks']:
                                img_stack = data['patch_stacks'][stack_number]['Image data']  # (H, W, C, N)
                                sharpness_array = data['patch_stacks'][stack_number]['sharpness']
                                max_sharpness_index = np.argmax(sharpness_array)

                                img_stack = np.transpose(img_stack, (3, 0, 1, 2))  # (N, H, W, C)
                                modified_images = []

                                for i in range(img_stack.shape[0]):
                                    image = img_stack[i].copy()
                                    sharpness_value = int(round(sharpness_array[i]))
                                    padded_image = cv2.copyMakeBorder(
                                        image, 0, 50, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                                    )
                                    cv2.putText(padded_image, f"{sharpness_value}", (30, 105),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                                    if i == max_sharpness_index:
                                        image = np.nan_to_num(padded_image, nan=0.0, posinf=255, neginf=0)
                                        if image.dtype != np.uint8:
                                            image = np.clip(image, 0, 255).astype(np.uint8)
                                        modified_images.append(image)
                                    else:
                                        modified_images.append(padded_image)

                                result = np.concatenate(modified_images, axis=1)
                                out_name = f"{stack_number}_{data['FOV number']}.png"
                                out_path = os.path.join(output_dir, out_name)
                                cv2.imwrite(out_path, result)
                                found_data = True
                                
                        except Exception as e:
                            print(f"Error processing {pkl_file}: {e}")
                            continue

    if found_data:
        print(f"[INFO] Process accession without scale completed: {output_dir}")
        return output_dir
    else:
        print(f"[WARNING] No data found for {accession_number}")
        return None


def process_accession_with_scale(idx, nats_path, base_dir):
    file_name = file_names[idx]
    accession_number = accession_numbers[idx]
    
    # Create output directory: base_dir/accession_number/with_scale
    output_dir = os.path.join(base_dir, accession_number, "with_scale")
    os.makedirs(output_dir, exist_ok=True)

    # Drawing parameters
    center_x, center_y = 40, 40
    y_line, x_line = 65, 70
    pixel_length = 3.21

    line_start = int(round(center_x - 5 * pixel_length))
    line_end = int(round(center_x + 5 * pixel_length))
    line_start_h = int(round(center_x - 3.5 * pixel_length))
    line_end_h = int(round(center_x + 3.5 * pixel_length))

    y_line_start = int(round(center_y - 5 * pixel_length))
    y_line_end = int(round(center_y + 5 * pixel_length))
    y_line_start_h = int(round(center_y - 3.5 * pixel_length))
    y_line_end_h = int(round(center_y + 3.5 * pixel_length))

    found_data = False

    for x in os.listdir(nats_path):
        if not os.path.isdir(os.path.join(nats_path, x)) or x == ".DS_Store":
            continue
        if x != accession_number:
            continue

        for date in os.listdir(os.path.join(nats_path, x)):
            date_path = os.path.join(nats_path, x, date)
            if not os.path.isdir(date_path):
                continue
                
            for tray_id in os.listdir(date_path):
                tray_path = os.path.join(date_path, tray_id)
                if not os.path.isdir(tray_path):
                    continue
                    
                for y in os.listdir(tray_path):
                    if y != file_name:
                        continue

                    data_path = os.path.join(tray_path, y)
                    pkl_stack_path = os.path.join(data_path, "wbc", "Recon_data", "pkl_stacks")
                    
                    if not os.path.exists(pkl_stack_path):
                        continue

                    pkl_files = [f for f in os.listdir(pkl_stack_path) if f.lower().endswith(".pkl")]
                    for pkl_file in pkl_files:
                        pkl_path = os.path.join(pkl_stack_path, pkl_file)
                        try:
                            with open(pkl_path, 'rb') as f:
                                data = pickle.load(f)

                            for stack_number in data['patch_stacks']:
                                img_stack = data['patch_stacks'][stack_number]['Image data']  # (H, W, C, N)
                                sharpness_array = data['patch_stacks'][stack_number]['sharpness']
                                max_sharpness_index = np.argmax(sharpness_array)

                                img_stack = np.transpose(img_stack, (3, 0, 1, 2))  # (N, H, W, C)
                                modified_images = []

                                for i in range(img_stack.shape[0]):
                                    image = img_stack[i].copy()
                                    sharpness_value = int(round(sharpness_array[i]))
                                    padded_image = cv2.copyMakeBorder(
                                        image, 0, 50, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                                    )
                                    cv2.putText(padded_image, f"{sharpness_value}", (30, 105),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                                    if i == max_sharpness_index:
                                        image = np.nan_to_num(padded_image, nan=0.0, posinf=255, neginf=0)
                                        if image.dtype != np.uint8:
                                            image = np.clip(image, 0, 255).astype(np.uint8)
                                        
                                        # Draw scale lines
                                        cv2.line(image, (line_start, y_line), (line_end, y_line), (0, 0, 0), 1)
                                        cv2.line(image, (line_start_h, y_line - 3), (line_start_h, y_line + 3), (0, 0, 0), 1)
                                        cv2.line(image, (line_end_h, y_line - 3), (line_end_h, y_line + 3), (0, 0, 0), 1)
                                        cv2.putText(image, "10", (line_end - 3, y_line + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
                                        cv2.putText(image, "7", (line_end_h - 3, y_line + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
                                        cv2.line(image, (line_start, y_line - 5), (line_start, y_line + 3), (0, 0, 0), 1)
                                        cv2.line(image, (line_end, y_line - 5), (line_end, y_line + 3), (0, 0, 0), 1)

                                        cv2.line(image, (x_line, y_line_start), (x_line, y_line_end), (0, 0, 0), 1)
                                        cv2.line(image, (x_line - 5, y_line_start), (x_line + 5, y_line_start), (0, 0, 0), 1)
                                        cv2.line(image, (x_line - 5, y_line_end), (x_line + 5, y_line_end), (0, 0, 0), 1)

                                        cv2.putText(image, "10", (x_line + 3, y_line_start + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
                                        cv2.putText(image, "7", (x_line + 3, y_line_start_h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
                                        cv2.line(image, (x_line - 3, y_line_start_h), (x_line + 3, y_line_start_h), (0, 0, 0), 1)
                                        cv2.line(image, (x_line - 3, y_line_end_h), (x_line + 3, y_line_end_h), (0, 0, 0), 1)
                                        
                                        modified_images.append(image)
                                    else:
                                        modified_images.append(padded_image)

                                result = np.concatenate(modified_images, axis=1)
                                out_name = f"{stack_number}_{data['FOV number']}.png"
                                out_path = os.path.join(output_dir, out_name)
                                cv2.imwrite(out_path, result)
                                found_data = True
                                
                        except Exception as e:
                            print(f"Error processing {pkl_file}: {e}")
                            continue

    if found_data:
        print(f"[INFO] Process accession with scale completed: {output_dir}")
        return output_dir
    else:
        print(f"[WARNING] No data found for {accession_number}")
        return None


def main(idx):
    # Mount SMB shares
    model_path = "/home/sigvet/strip creation/wbc_tc_dc_iter_6.onnx"
    public_mount = mount_smb_share("Public", "/mnt/Public")
    sigvet_mount = mount_smb_share("Sigvet_DC", "/mnt/Sigvet_DC")

    # Define paths
    nats_path = os.path.join(public_mount, "Sigvet_DC", "PD10", "448771007")
    base_dir = os.path.join(sigvet_mount, "PD10_1")  # Consistent case
    
    print(f"[INFO] Processing index {idx}: {accession_numbers[idx]}")

    # Process both with and without scale
    des_path_with_scale = process_accession_with_scale(idx, nats_path, base_dir)
    des_path_without_scale = process_accession_without_scale(idx, nats_path, base_dir)
    
    if des_path_without_scale and os.path.exists(des_path_without_scale):
        # Extract patches from "without scale" results
        patch_path = patch_extraction(des_path_without_scale)
        
        if patch_path and os.path.exists(patch_path):
            # Create classify directory path
            parent_dir = os.path.dirname(patch_path)
            output_path_classify = os.path.join(parent_dir, "classify")
            
            # Classify patches
            try:
                classify(patch_path, output_path_classify, model_path)
                
                # Create scaled strips in classes folder
                if des_path_with_scale and os.path.exists(des_path_with_scale):
                    scaled_strips_in_classes_folder(output_path_classify, des_path_with_scale)
                else:
                    print(f"[WARNING] With scale directory not found for {accession_numbers[idx]}")
                    
            except Exception as e:
                print(f"[ERROR] Classification failed for {accession_numbers[idx]}: {e}")
        else:
            print(f"[ERROR] Patch extraction failed for {accession_numbers[idx]}")
    else:
        print(f"[ERROR] Without scale processing failed for {accession_numbers[idx]}")


# Main execution
if __name__ == "__main__":
    for idx in range(len(file_names)):
        try:
            main(idx)
        except Exception as e:
            print(f"[ERROR] Failed processing index {idx} ({accession_numbers[idx]}): {e}")
            continue
    
    print("[INFO] All processing completed!")