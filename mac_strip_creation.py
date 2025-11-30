import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import subprocess

# ---------------- SMB MOUNT ----------------
def mount_smb_share(share_name, mount_point):
    server = "192.168.0.253"
    username = "Sigtuple"
    password = "Sigtuple@123"

    if not os.path.ismount(mount_point):
        print(f"Mounting {server}/{share_name} at {mount_point} ...")
        os.makedirs(mount_point, exist_ok=True)
        smb_url = f"//{username}:{password}@{server}/{share_name}"
        try:
            subprocess.run(["mount_smbfs", smb_url, mount_point], check=True)
            print(f"✅ Mounted {share_name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to mount {share_name}:", e)
            raise
    else:
        print(f"✅ Already mounted: {mount_point}")
    return mount_point

# Mount both shares
public_mount = mount_smb_share("Public", "/Volumes/Public")
sigvet_mount = mount_smb_share("Sigvet_DC", "/Volumes/Sigvet_DC")

# ---------------- CSV ----------------
csv_file_path = '/Users/benjaminroyv/sigvet/strip creation /72 samples_residuals - combined_patient_data.csv'
csv_file = pd.read_csv(csv_file_path)

file_names = csv_file['Order ID_PD10'].tolist()
accession_numbers = csv_file['Patient ID'].tolist()

# ---------------- PROCESS FUNCTION ----------------
def process_accession(idx):
    file_name = file_names[idx]
    accession_number = accession_numbers[idx]

    # Input folder in Public share
    nats_path = os.path.join(public_mount, "Sigvet_DC", "PD10", "448771007")

    if not os.path.exists(nats_path):
        print(f"❌ Source path does not exist: {nats_path}")
        return accession_number

    # Destination folder in Sigvet_DC share
    des_path = os.path.join(sigvet_mount, "pd10")
    os.makedirs(des_path, exist_ok=True)

    # Drawing / patch parameters
    center_x, center_y, y_line, x_line, pixel_length = 40, 40, 65, 70, 3.21
    line_start = int(round(center_x - 5 * pixel_length))
    line_end = int(round(center_x + 5 * pixel_length))
    line_start_h = int(round(center_x - 3.5 * pixel_length))
    line_end_h = int(round(center_x + 3.5 * pixel_length))
    y_line_start = int(round(center_y - 5 * pixel_length))
    y_line_end = int(round(center_y + 5 * pixel_length))
    y_line_start_h = int(round(center_y - 3.5 * pixel_length))
    y_line_end_h = int(round(center_y + 3.5 * pixel_length))

    # Traverse folders
    for x in os.listdir(nats_path):
        if x.startswith(".") or not os.path.isdir(os.path.join(nats_path, x)):
            continue
        if x != accession_number:
            continue
        print(f"Processing accession number: {x}")

        accession_path = os.path.join(nats_path, x)
        for date in os.listdir(accession_path):
            for tray_id in os.listdir(os.path.join(accession_path, date)):
                for y in os.listdir(os.path.join(accession_path, date, tray_id)):
                    if y != file_name:
                        continue
                    data_path = os.path.join(accession_path, date, tray_id, y)
                    des_path_full = os.path.join(des_path, accession_number, date, file_name)
                    os.makedirs(des_path_full, exist_ok=True)

                    # PKL stacks
                    pkl_stack_path = os.path.join(data_path, "wbc", "Recon_data", "pkl_stacks")
                    if not os.path.exists(pkl_stack_path):
                        print(f"❌ PKL path missing: {pkl_stack_path}")
                        return accession_number

                    pkl_files = [f for f in os.listdir(pkl_stack_path) if f.lower().endswith(".pkl")]
                    for pkl_file in pkl_files:
                        pkl_path = os.path.join(pkl_stack_path, pkl_file)
                        with open(pkl_path, 'rb') as f:
                            data = pickle.load(f)

                        for stack_number in data['patch_stacks']:
                            img_stack = data['patch_stacks'][stack_number]['Image data']  # (H,W,C,N)
                            sharpness_array = data['patch_stacks'][stack_number]['sharpness']
                            max_sharpness_index = np.argmax(sharpness_array)

                            img_stack = np.transpose(img_stack, (3,0,1,2))  # (N,H,W,C)
                            modified_images = []

                            for i in range(img_stack.shape[0]):
                                image = img_stack[i].copy()
                                sharpness_value = int(round(sharpness_array[i]))
                                padded_image = cv2.copyMakeBorder(
                                    image, 0, 50, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0)
                                )
                                cv2.putText(padded_image, f"{sharpness_value}", (30,105),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                                if i == max_sharpness_index:
                                    image = np.nan_to_num(padded_image, nan=0.0, posinf=255, neginf=0)
                                    if image.dtype != np.uint8:
                                        image = np.clip(image,0,255).astype(np.uint8)
                                    modified_images.append(image)
                                else:
                                    modified_images.append(padded_image)

                            result = np.concatenate(modified_images, axis=1)
                            out_name = f"{stack_number}_{data['FOV number']}.png"
                            out_path = os.path.join(des_path_full, out_name)
                            cv2.imwrite(out_path, result)

    # ZIP the processed accession
    accession_folder = os.path.join(des_path, accession_number)
    if os.path.exists(accession_folder):
        shutil.make_archive(accession_folder, 'zip', accession_folder)
        print(f"✅ Compressed: {accession_number}.zip")

    return accession_number

# ---------------- MULTITHREADING ----------------
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_accession, idx) for idx in range(len(file_names))]
    for f in tqdm(as_completed(futures), total=len(futures), desc="Processing all accessions"):
        try:
            result = f.result()
            print(f"Finished: {result}")
        except Exception as e:
            print(f"Error: {e}")
