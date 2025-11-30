import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil


csv_file = pd.read_csv('/Users/benjaminroyv/sigvet/strip creation /72 samples_residuals - combined_patient_data.csv')

file_names = []
accession_numbers = []

# Collect file names and accession numbers
for index, row in csv_file.iterrows():
    file_names.append(row['Order ID_PD10'])
    accession_numbers.append(row['Patient ID'])


def process_accession_with_scale(idx):
    """
    Process a single accession number + file_name (with scale).
    """
    file_name = file_names[idx]
    accession_number = accession_numbers[idx]

    nats_path = "//192.168.0.253/Public/Sigvet_DC/PD10/448771007"
    des_path = "//192.168.0.253/Micro-POC/Sigvet_MicroPOC/SigVet_DS/September_data/PD10/with_scale"
    os.makedirs(des_path, exist_ok=True)

    # Drawing parameters
    center_x, center_y = 40, 40
    y_line, x_line = 65, 70
    pixel_length = 3.21

    line_start = int(round(center_x - 5 * pixel_length))
    line_end   = int(round(center_x + 5 * pixel_length))
    line_start_h = int(round(center_x - 3.5 * pixel_length))
    line_end_h   = int(round(center_x + 3.5 * pixel_length))

    y_line_start = int(round(center_y - 5 * pixel_length))
    y_line_end   = int(round(center_y + 5 * pixel_length))
    y_line_start_h = int(round(center_y - 3.5 * pixel_length))
    y_line_end_h   = int(round(center_y + 3.5 * pixel_length))

    # Traverse folders
    for x in os.listdir(nats_path):
        if not os.path.isdir(os.path.join(nats_path, x)) or x == ".DS_Store":
            continue
        if x != accession_number:
            continue
        else:
            print("Processing accession number (with scale):", x)

        for date in os.listdir(os.path.join(nats_path, x)):
            for tray_id in os.listdir(os.path.join(nats_path, x, date)):
                for y in os.listdir(os.path.join(nats_path, x, date, tray_id)):
                    if y == file_name:
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
                                        # horizontal scale
                                        cv2.line(image, (line_start, y_line), (line_end, y_line), (0, 0, 0), 1)
                                        cv2.line(image, (line_start_h, y_line - 3), (line_start_h, y_line + 3), (0, 0, 0), 1)
                                        cv2.line(image, (line_end_h, y_line - 3), (line_end_h, y_line + 3), (0, 0, 0), 1)
                                        cv2.putText(image, "10", (line_end - 3, y_line + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
                                        cv2.putText(image, "7", (line_end_h - 3, y_line + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
                                        cv2.line(image, (line_start, y_line - 5), (line_start, y_line + 3), (0, 0, 0), 1)
                                        cv2.line(image, (line_end, y_line - 5), (line_end, y_line + 3), (0, 0, 0), 1)

                                        # vertical scale
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
                                out_path = os.path.join(des_path_full, out_name)
                                cv2.imwrite(out_path, result)

    return accession_number


def process_accession_without(idx):
    """
    Process a single accession number + file_name (without scale).
    """
    file_name = file_names[idx]
    accession_number = accession_numbers[idx]

    nats_path = "//192.168.0.253/Public/Sigvet_DC/PD10/448771007"
    des_path = "//192.168.0.253/Micro-POC/Sigvet_MicroPOC/SigVet_DS/September_data/PD10/without_scale"
    os.makedirs(des_path, exist_ok=True)

    # Traverse folders
    for x in os.listdir(nats_path):
        if not os.path.isdir(os.path.join(nats_path, x)) or x == ".DS_Store":
            continue
        if x != accession_number:
            continue
        else:
            print("Processing accession number (without scale):", x)

        for date in os.listdir(os.path.join(nats_path, x)):
            for tray_id in os.listdir(os.path.join(nats_path, x, date)):
                for y in os.listdir(os.path.join(nats_path, x, date, tray_id)):
                    if y == file_name:
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

    for i in range(0, (len_z_stack)):
        img = image[0:patch_h, i * patch_w: patch_w + i * patch_w]
        sharpness = get_image_sharpness(img)
        img_count_sharpness_pair.append((i, sharpness))

    sorted_data = sorted(img_count_sharpness_pair, key=lambda x: x[1])
    max_index = len(sorted_data) - 1
    max_count = sorted_data[max_index][0]
    input_patch = image[0:patch_h, max_count * patch_w: patch_w + max_count * patch_w]

    return input_patch


def patch_extraction(input_dir):
    output_dir = input_dir.replace("without_scale", "patches")
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for image_file in files:
            if not image_file.endswith(".png"):
                continue
            image_path = os.path.join(root, image_file)
            image = cv2.imread(image_path)
            if image is None:
                continue
            patch = extract_input_patch(image)
            out_path = os.path.join(output_dir, image_file)
            cv2.imwrite(out_path, patch)

    return output_dir


# ================= MAIN ==================
if __name__ == "__main__":
    results_without = []

    # -------- Run WITH SCALE --------
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     futures = [executor.submit(process_accession_with_scale, idx) for idx in range(len(file_names))]
    #     for f in tqdm(as_completed(futures), total=len(futures), desc="Processing with scale"):
    #         try:
    #             result = f.result()
    #             if result:
    #                 print(f"With scale finished: {result}")
    #         except Exception as e:
    #             print(f"Error (with scale): {e}")

    # -------- Run WITHOUT SCALE --------
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_accession_without, idx) for idx in range(len(file_names))]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing without scale"):
            try:
                result = f.result()
                if result:
                    results_without.append(result)
                    print(f"Without scale finished: {result}")
            except Exception as e:
                print(f"Error (without scale): {e}")

    # -------- Patch extraction on WITHOUT results --------
    for des_path in results_without:
        try:
            patch_path = patch_extraction(des_path)
            print(f"Patches extracted at: {patch_path}")
        except Exception as e:
            print(f"Error during patch extraction for {des_path}: {e}")
