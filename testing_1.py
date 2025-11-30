import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# ---------- Mount SMB Share (Linux) ----------
def mount_smb_share():
    mount_point = "/mnt/sigvet"
    server = "192.168.0.253"
    share = "Sigvet_DC"
    username = "Sigtuple"
    password = "Sigtuple@123"

    if not os.path.ismount(mount_point):
        os.makedirs(mount_point, exist_ok=True)
        cmd = [
            "sudo", "mount", "-t", "cifs",
            f"//{server}/{share}", mount_point,
            "-o", f"username={username},password={password},vers=3.0,rw"
        ]
        subprocess.run(cmd, check=True)
    
    return mount_point

# ---------- Global Setup ----------
zip_dir = "/mnt/sigvet/pd10/zip"
csv_file = pd.read_csv('/Users/benjaminroyv/sigvet/strip creation /72 samples_residuals - combined_patient_data.csv')

file_names = csv_file['Order ID_PD10'].tolist()
accession_numbers = csv_file['Patient ID'].tolist()

# ---------- Helpers ----------
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
    output_dir = input_dir.replace("without", "patches")
    os.makedirs(output_dir, exist_ok=True)

    for image_file in os.listdir(input_dir):
        if image_file == ".DS_Store":
            continue
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        patch = extract_input_patch(image)
        cv2.imwrite(os.path.join(output_dir, image_file), patch)

    return output_dir

# ---------- Main Processing Functions ----------
def process_accession_with_scale(idx, nats_path, des_path):
    file_name = file_names[idx]
    accession_number = accession_numbers[idx]

    os.makedirs(des_path, exist_ok=True)

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

    for x in os.listdir(nats_path):
        if not os.path.isdir(os.path.join(nats_path, x)) or x == ".DS_Store":
            continue
        if x != accession_number:
            continue
        print("Processing accession number (with scale):", x)

        for date in os.listdir(os.path.join(nats_path, x)):
            for tray_id in os.listdir(os.path.join(nats_path, x, date)):
                for y in os.listdir(os.path.join(nats_path, x, date, tray_id)):
                    if y != file_name:
                        continue

                    data_path = os.path.join(nats_path, x, date, tray_id, y)
                    des_path_full = os.path.join(des_path, accession_number, date, file_name)
                    os.makedirs(des_path_full, exist_ok=True)

                    pkl_stack_path = os.path.join(data_path, "wbc", "Recon_data", "pkl_stacks")
                    if not os.path.exists(pkl_stack_path):
                        print(f"Path does not exist: {pkl_stack_path}")
                        return accession_number

                    pkl_files = [f for f in os.listdir(pkl_stack_path) if f.lower().endswith(".pkl")]
                    for pkl_file in pkl_files:
                        pkl_path = os.path.join(pkl_stack_path, pkl_file)
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
                                    cv2.line(image, (line_start, y_line), (line_end, y_line), (0, 0, 0), 1)
                                    cv2.line(image, (line_start_h, y_line - 3), (line_start_h, y_line + 3), (0, 0, 0), 1)
                                    cv2.line(image, (line_end_h, y_line - 3), (line_end_h, y_line + 3), (0, 0, 0), 1)
                                    modified_images.append(image)
                                else:
                                    modified_images.append(padded_image)

                            result = np.concatenate(modified_images, axis=1)
                            out_name = f"{stack_number}_{data['FOV number']}.png"
                            out_path = os.path.join(des_path_full, out_name)
                            cv2.imwrite(out_path, result)

    return des_path


def process_accession_without_scale(idx, nats_path, des_path):
    file_name = file_names[idx]
    accession_number = accession_numbers[idx]

    os.makedirs(des_path, exist_ok=True)

    for x in os.listdir(nats_path):
        if not os.path.isdir(os.path.join(nats_path, x)) or x == ".DS_Store":
            continue
        if x != accession_number:
            continue
        print("Processing accession number (without scale):", x)

        for date in os.listdir(os.path.join(nats_path, x)):
            for tray_id in os.listdir(os.path.join(nats_path, x, date)):
                for y in os.listdir(os.path.join(nats_path, x, date, tray_id)):
                    if y != file_name:
                        continue

                    data_path = os.path.join(nats_path, x, date, tray_id, y)
                    des_path_full = os.path.join(des_path, accession_number, date, file_name)
                    os.makedirs(des_path_full, exist_ok=True)

                    pkl_stack_path = os.path.join(data_path, "wbc", "Recon_data", "pkl_stacks")
                    if not os.path.exists(pkl_stack_path):
                        print(f"Path does not exist: {pkl_stack_path}")
                        return accession_number

                    pkl_files = [f for f in os.listdir(pkl_stack_path) if f.lower().endswith(".pkl")]
                    for pkl_file in pkl_files:
                        pkl_path = os.path.join(pkl_stack_path, pkl_file)
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
                            out_path = os.path.join(des_path_full, out_name)
                            cv2.imwrite(out_path, result)

    return des_path

# ---------- Main ----------
if __name__ == "__main__":
    # Mount SMB share
    base_mount = mount_smb_share()

    # Define paths inside mounted share
    nats_path = os.path.join(base_mount, "PD10", "448771007")
    des_with_scale = os.path.join(base_mount, "output_with_scale")
    des_without_scale = os.path.join(base_mount, "output_without_scale")

    # Run one sample for testing
    process_accession_with_scale(0, nats_path, des_with_scale)
    des_path_no_scale = process_accession_without_scale(0, nats_path, des_without_scale)

    # Extract patches from "without scale" results
    patch_path = patch_extraction(des_path_no_scale)
    print(f"Patches saved to: {patch_path}")
