# import os
# import subprocess
# from google.cloud import storage
# from google.oauth2 import service_account
# import zipfile
# import pickle
# import numpy as np
# import cv2
# import shutil
# import time


# def mount_smb_share(share_name, mount_point):
#     server = "192.168.0.253"
#     username = "Sigtuple"
#     password = "Sigtuple@123"

#     if not os.path.ismount(mount_point):
#         print(f"Mounting {server}/{share_name} at {mount_point} ...")
#         os.makedirs(mount_point, exist_ok=True)
#         smb_url = f"//{username}:{password}@{server}/{share_name}"
#         try:
#             subprocess.run(["mount_smbfs", smb_url, mount_point], check=True)
#             print(f"✅ Mounted {share_name}")
#         except subprocess.CalledProcessError as e:
#             print(f"❌ Failed to mount {share_name}: {e}")
#             raise
#     else:
#         print(f"✅ Already mounted: {mount_point}")
#     return mount_point

# def process_accession_with_scale(file_path, des_path):
#     # Create output directory: base_dir/accession_number/with_scale

#     output_dir = os.path.join(des_path, "without_scale")
#     os.makedirs(output_dir, exist_ok=True)

#     # Drawing parameters
#     center_x, center_y = 40, 40
#     y_line, x_line = 65, 70
#     pixel_length = 3.21

#     line_start = int(round(center_x - 5 * pixel_length))
#     line_end = int(round(center_x + 5 * pixel_length))
#     line_start_h = int(round(center_x - 3.5 * pixel_length))
#     line_end_h = int(round(center_x + 3.5 * pixel_length))

#     y_line_start = int(round(center_y - 5 * pixel_length))
#     y_line_end = int(round(center_y + 5 * pixel_length))
#     y_line_start_h = int(round(center_y - 3.5 * pixel_length))
#     y_line_end_h = int(round(center_y + 3.5 * pixel_length))

#     found_data = False

#     # for x in os.listdir(file_path):
#     #     if not os.path.isdir(os.path.join(file_path, x)) or x == ".DS_Store":
#     #         continue
#     #     if x != accession_number:
#     #         continue

#     #     for date in os.listdir(os.path.join(file_path, x)):
#     #         date_path = os.path.join(file_path, x, date)
#     #         if not os.path.isdir(date_path):
#     #             continue
                
#     #         for tray_id in os.listdir(date_path):
#     #             tray_path = os.path.join(date_path, tray_id)
#     #             if not os.path.isdir(tray_path):
#     #                 continue
                    
#     #             for y in os.listdir(tray_path):
#     #                 if y != file_name:
#     #                     continue

#     #                 data_path = os.path.join(tray_path, y)
#     pkl_stack_path = os.path.join(file_path, "wbc", "Recon_data", "pkl_stacks")
#     # pkl_stack_path= file_path
    
#     if not os.path.exists(pkl_stack_path):
#         print(f"[WARNING] Path does not exist: {pkl_stack_path}")
#         raise FileNotFoundError(f"Path does not exist: {pkl_stack_path}")

#     pkl_files = [f for f in os.listdir(pkl_stack_path) if f.lower().endswith(".pkl")]
#     for pkl_file in pkl_files:
#         pkl_path = os.path.join(pkl_stack_path, pkl_file)
#         with open(pkl_path, 'rb') as f:
#             data = pickle.load(f)

#         for stack_number in data['patch_stacks']:
#             img_stack = data['patch_stacks'][stack_number]['Image data']  # (H, W, C, N)
#             sharpness_array = data['patch_stacks'][stack_number]['sharpness']
#             max_sharpness_index = np.argmax(sharpness_array)

#             img_stack = np.transpose(img_stack, (3, 0, 1, 2))  # (N, H, W, C)
#             modified_images = []

#             for i in range(img_stack.shape[0]):
#                 image = img_stack[i].copy()
#                 sharpness_value = int(round(sharpness_array[i]))
#                 padded_image = cv2.copyMakeBorder(
#                     image, 0, 50, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
#                 )
#                 cv2.putText(padded_image, f"{sharpness_value}", (30, 105),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

#                 if i == max_sharpness_index:
#                     image = np.nan_to_num(padded_image, nan=0.0, posinf=255, neginf=0)
#                     if image.dtype != np.uint8:
#                         image = np.clip(image, 0, 255).astype(np.uint8)
                    
#                     # # Draw scale lines
#                     # cv2.line(image, (line_start, y_line), (line_end, y_line), (0, 0, 0), 1)
#                     # cv2.line(image, (line_start_h, y_line - 3), (line_start_h, y_line + 3), (0, 0, 0), 1)
#                     # cv2.line(image, (line_end_h, y_line - 3), (line_end_h, y_line + 3), (0, 0, 0), 1)
#                     # cv2.putText(image, "10", (line_end - 3, y_line + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
#                     # cv2.putText(image, "7", (line_end_h - 3, y_line + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
#                     # cv2.line(image, (line_start, y_line - 5), (line_start, y_line + 3), (0, 0, 0), 1)
#                     # cv2.line(image, (line_end, y_line - 5), (line_end, y_line + 3), (0, 0, 0), 1)

#                     # cv2.line(image, (x_line, y_line_start), (x_line, y_line_end), (0, 0, 0), 1)
#                     # cv2.line(image, (x_line - 5, y_line_start), (x_line + 5, y_line_start), (0, 0, 0), 1)
#                     # cv2.line(image, (x_line - 5, y_line_end), (x_line + 5, y_line_end), (0, 0, 0), 1)

#                     # cv2.putText(image, "10", (x_line + 3, y_line_start + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
#                     # cv2.putText(image, "7", (x_line + 3, y_line_start_h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
#                     # cv2.line(image, (x_line - 3, y_line_start_h), (x_line + 3, y_line_start_h), (0, 0, 0), 1)
#                     # cv2.line(image, (x_line - 3, y_line_end_h), (x_line + 3, y_line_end_h), (0, 0, 0), 1)
                    
#                     modified_images.append(image)
#                 else:
#                     modified_images.append(padded_image)

#             result = np.concatenate(modified_images, axis=1)
#             out_name = f"{stack_number}_{data['FOV number']}.png"
#             out_path = os.path.join(output_dir, out_name)
#             cv2.imwrite(out_path, result)
#             found_data = True


#     output_dir = os.path.join(des_path, "with_scale")
#     os.makedirs(output_dir, exist_ok=True)

#     pkl_stack_path = os.path.join(file_path, "wbc", "Recon_data", "pkl_stacks")
#     # pkl_stack_path= file_path
    
#     if not os.path.exists(pkl_stack_path):
#         print(f"[WARNING] Path does not exist: {pkl_stack_path}")
#         raise FileNotFoundError(f"Path does not exist: {pkl_stack_path}")

#     pkl_files = [f for f in os.listdir(pkl_stack_path) if f.lower().endswith(".pkl")]
#     for pkl_file in pkl_files:
#         pkl_path = os.path.join(pkl_stack_path, pkl_file)
#         with open(pkl_path, 'rb') as f:
#             data = pickle.load(f)

#         for stack_number in data['patch_stacks']:
#             img_stack = data['patch_stacks'][stack_number]['Image data']  # (H, W, C, N)
#             sharpness_array = data['patch_stacks'][stack_number]['sharpness']
#             max_sharpness_index = np.argmax(sharpness_array)

#             img_stack = np.transpose(img_stack, (3, 0, 1, 2))  # (N, H, W, C)
#             modified_images = []

#             for i in range(img_stack.shape[0]):
#                 image = img_stack[i].copy()
#                 sharpness_value = int(round(sharpness_array[i]))
#                 padded_image = cv2.copyMakeBorder(
#                     image, 0, 50, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
#                 )
#                 cv2.putText(padded_image, f"{sharpness_value}", (30, 105),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

#                 if i == max_sharpness_index:
#                     image = np.nan_to_num(padded_image, nan=0.0, posinf=255, neginf=0)
#                     if image.dtype != np.uint8:
#                         image = np.clip(image, 0, 255).astype(np.uint8)
                    
#                     # # Draw scale lines
#                     cv2.line(image, (line_start, y_line), (line_end, y_line), (0, 0, 0), 1)
#                     cv2.line(image, (line_start_h, y_line - 3), (line_start_h, y_line + 3), (0, 0, 0), 1)
#                     cv2.line(image, (line_end_h, y_line - 3), (line_end_h, y_line + 3), (0, 0, 0), 1)
#                     cv2.putText(image, "10", (line_end - 3, y_line + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
#                     cv2.putText(image, "7", (line_end_h - 3, y_line + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
#                     cv2.line(image, (line_start, y_line - 5), (line_start, y_line + 3), (0, 0, 0), 1)
#                     cv2.line(image, (line_end, y_line - 5), (line_end, y_line + 3), (0, 0, 0), 1)

#                     cv2.line(image, (x_line, y_line_start), (x_line, y_line_end), (0, 0, 0), 1)
#                     cv2.line(image, (x_line - 5, y_line_start), (x_line + 5, y_line_start), (0, 0, 0), 1)
#                     cv2.line(image, (x_line - 5, y_line_end), (x_line + 5, y_line_end), (0, 0, 0), 1)

#                     cv2.putText(image, "10", (x_line + 3, y_line_start + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
#                     cv2.putText(image, "7", (x_line + 3, y_line_start_h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
#                     cv2.line(image, (x_line - 3, y_line_start_h), (x_line + 3, y_line_start_h), (0, 0, 0), 1)
#                     cv2.line(image, (x_line - 3, y_line_end_h), (x_line + 3, y_line_end_h), (0, 0, 0), 1)
                    
#                     modified_images.append(image)
#                 else:
#                     modified_images.append(padded_image)

#             result = np.concatenate(modified_images, axis=1)
#             out_name = f"{stack_number}_{data['FOV number']}.png"
#             out_path = os.path.join(output_dir, out_name)
#             cv2.imwrite(out_path, result)
#             found_data = True

#     if found_data:
#         print(f"[INFO] Process accession with scale completed: {output_dir}")
#         return output_dir
#     else:
#         print(f"[WARNING] No data found for {accession_number}")
#         return None



# sigvet_mount = mount_smb_share("Sigvet_DC", "/Volumes/Sigvet_DC")
# nas_path = os.path.join(sigvet_mount, "equine_data")
# local_path = "/Users/benjaminroyv/sigvet/strip creation"
# os.makedirs(os.path.dirname(local_path), exist_ok=True)
# target_files = [("1cb2d141-f074-4191-b237-533b9019f0f7","bb1d616a-282e-4882-806b-cc596a58cb28")]
# gcp_folder_name = "SIG-SIGVET-DC-05"
# extract_path = os.path.join(local_path, "extracted")
# os.makedirs(extract_path, exist_ok=True)
# strip_path = os.path.join(local_path, "strips")
# os.makedirs(strip_path, exist_ok=True)

# key_path = "/Users/benjaminroyv/sigvet/gcp_data/skilled-bonus-152013-5339881607a7.json"
# credentials = service_account.Credentials.from_service_account_file(key_path)
# gcp_client = storage.Client(credentials=credentials, project=credentials.project_id)
# bucket = gcp_client.bucket("sigvet_data")

# blobs = list(gcp_client.list_blobs("sigvet_data", prefix=gcp_folder_name))
# gcp_folder_names = []

# for tray_run_id, order_id in target_files:
#     for blob in blobs:
#         zip_folder_path = blob.name
#         zip_folder_name = zip_folder_path.split("/")[-1]
#         zip_folder_name_only_uuid = zip_folder_name.split(".zip")[0]

#         if tray_run_id == zip_folder_name_only_uuid:
#             os.makedirs(os.path.dirname(os.path.join(local_path, zip_folder_name_only_uuid)), exist_ok=True)
#             print(f"⬇️ Downloading {zip_folder_name} ...")
#             zip_file_name = os.path.join(local_path, zip_folder_name)
#             with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
#                 zip_ref.extractall(extract_path)
#             full_path = os.path.join(extract_path, zip_folder_name_only_uuid,order_id)
#             des_path = os.path.join(strip_path, zip_folder_name_only_uuid, order_id)
#             os.makedirs(des_path, exist_ok=True)
#             process_accession_with_scale(full_path, des_path)
#             blob.download_to_filename(os.path.join(local_path, zip_folder_name))

#             print(f"✅ Downloaded {zip_folder_name} → {os.path.join(local_path, zip_folder_name_only_uuid)}")
#             gcp_folder_names.append(zip_folder_name_only_uuid)

# print("Matched & downloaded:", gcp_folder_names)


import os
import subprocess
from google.cloud import storage
from google.oauth2 import service_account
import zipfile
import pickle
import numpy as np
import cv2
import shutil
import time

def mount_smb_share(share_name, mount_point, server="192.168.0.253", username="Sigtuple", password="Sigtuple@123"):
    """
    Mounts an SMB share to a local directory if not already mounted.
    """
    if not os.path.ismount(mount_point):
        print(f"Mounting {server}/{share_name} at {mount_point} ...")
        os.makedirs(mount_point, exist_ok=True)
        smb_url = f"//{username}:{password}@{server}/{share_name}"
        try:
            subprocess.run(["mount_smbfs", smb_url, mount_point], check=True, capture_output=True, text=True)
            print(f"✅ Mounted {share_name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to mount {share_name}: {e.stderr}")
            raise
    else:
        print(f"✅ Already mounted: {mount_point}")
    return mount_point

def process_images_from_pkl(pkl_stack_path, output_dir, draw_scale):
    """
    Processes image stacks from pickle files, draws sharpness values,
    optionally draws a scale bar, and saves the resulting image strips.
    """
    os.makedirs(output_dir, exist_ok=True)
    found_data = False

    if not os.path.exists(pkl_stack_path):
        print(f"[WARNING] Path does not exist: {pkl_stack_path}")
        return False

    # Drawing parameters
    center_x, center_y = 40, 40
    pixel_length = 3.21
    y_line, x_line = 65, 70

    # Horizontal scale bar coordinates
    line_start = int(round(center_x - 5 * pixel_length))
    line_end = int(round(center_x + 5 * pixel_length))
    line_start_h = int(round(center_x - 3.5 * pixel_length))
    line_end_h = int(round(center_x + 3.5 * pixel_length))

    # Vertical scale bar coordinates
    y_line_start = int(round(center_y - 5 * pixel_length))
    y_line_end = int(round(center_y + 5 * pixel_length))
    y_line_start_h = int(round(center_y - 3.5 * pixel_length))
    y_line_end_h = int(round(center_y + 3.5 * pixel_length))

    pkl_files = [f for f in os.listdir(pkl_stack_path) if f.lower().endswith(".pkl")]
    for pkl_file in pkl_files:
        pkl_path = os.path.join(pkl_stack_path, pkl_file)
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        for stack_number, stack_data in data.get('patch_stacks', {}).items():
            img_stack = stack_data.get('Image data')
            sharpness_array = stack_data.get('sharpness')

            if img_stack is None or sharpness_array is None:
                continue

            found_data = True
            max_sharpness_index = np.argmax(sharpness_array)
            img_stack_transposed = np.transpose(img_stack, (3, 0, 1, 2))  # (N, H, W, C)
            modified_images = []

            for i, image_slice in enumerate(img_stack_transposed):
                image = image_slice.copy()
                sharpness_value = int(round(sharpness_array[i]))
                
                # Add a black border at the bottom for text
                padded_image = cv2.copyMakeBorder(
                    image, 0, 50, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
                
                # Write sharpness value
                cv2.putText(padded_image, f"{sharpness_value}", (30, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                if i == max_sharpness_index:
                    # Clean up potential NaN/inf values
                    safe_image = np.nan_to_num(padded_image, nan=0.0, posinf=255, neginf=0)
                    if safe_image.dtype != np.uint8:
                        safe_image = np.clip(safe_image, 0, 255).astype(np.uint8)
                    
                    if draw_scale:
                        # Draw horizontal scale lines
                        cv2.line(safe_image, (line_start, y_line), (line_end, y_line), (0, 0, 0), 1)
                        cv2.line(safe_image, (line_start_h, y_line - 3), (line_start_h, y_line + 3), (0, 0, 0), 1)
                        cv2.line(safe_image, (line_end_h, y_line - 3), (line_end_h, y_line + 3), (0, 0, 0), 1)
                        cv2.putText(safe_image, "10", (line_end - 10, y_line + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(safe_image, "7", (line_end_h - 10, y_line + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)

                        # Draw vertical scale lines
                        cv2.line(safe_image, (x_line, y_line_start), (x_line, y_line_end), (0, 0, 0), 1)
                        cv2.line(safe_image, (x_line - 5, y_line_start), (x_line + 5, y_line_start), (0, 0, 0), 1)
                        cv2.line(safe_image, (x_line - 5, y_line_end), (x_line + 5, y_line_end), (0, 0, 0), 1)
                        cv2.putText(safe_image, "10", (x_line + 5, y_line_end - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(safe_image, "7", (x_line + 5, y_line_end_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
                    
                    modified_images.append(safe_image)
                else:
                    modified_images.append(padded_image)

            # Concatenate images into a single strip
            result = np.concatenate(modified_images, axis=1)
            out_name = f"{stack_number}_{data.get('FOV number', 'N/A')}.png"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, result)

    return found_data

def main():
    """
    Main execution function.
    """
    # --- Configuration ---
    local_path = "/Users/benjaminroyv/sigvet/strip_creation"
    key_path = "/Users/benjaminroyv/sigvet/gcp_data/skilled-bonus-152013-5339881607a7.json"
    gcp_bucket_name = "sigvet_data"
    gcp_folder_name = "SIGALP-N001-SV-0004"
    target_files = [
  ("c57654ee-8d89-4d4a-aa8f-5d173a69abea", "ef5af20d-e922-4f1a-add1-343b71731368"),
]
    
    # --- Setup local directories ---
    os.makedirs(local_path, exist_ok=True)
    extract_path = os.path.join(local_path, "extracted")
    sigvet_mount = mount_smb_share("Sigvet_DC", "/Volumes/Sigvet_DC")
    strip_path = sigvet_mount + "/exp"
    # strip_path = os.path.join(local_path, "strips")
    os.makedirs(extract_path, exist_ok=True)
    os.makedirs(strip_path, exist_ok=True)

    # --- Mount SMB Share (optional, if needed for source data) ---
    # sigvet_mount = mount_smb_share("Sigvet_DC", "/Volumes/Sigvet_DC")
    # nas_path = os.path.join(sigvet_mount, "equine_data")

    # --- Initialize Google Cloud client ---
    try:
        credentials = service_account.Credentials.from_service_account_file(key_path)
        gcp_client = storage.Client(credentials=credentials, project=credentials.project_id)
        bucket = gcp_client.bucket(gcp_bucket_name)
    except Exception as e:
        print(f"❌ Failed to initialize Google Cloud client: {e}")
        return

    # --- Process target files ---
    blobs = list(gcp_client.list_blobs(bucket, prefix=gcp_folder_name))
    downloaded_files = []

    for tray_run_id, order_id in target_files:
        found_blob = False
        for blob in blobs:
            if tray_run_id in blob.name:
                found_blob = True
                zip_file_name = os.path.basename(blob.name)
                local_zip_path = os.path.join(local_path, zip_file_name)
                
                print(f"⬇️ Downloading {zip_file_name} ...")
                try:
                    blob.download_to_filename(local_zip_path)
                    print(f"✅ Downloaded {zip_file_name}")
                    
                    # --- Unzip and Process ---
                    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                    
                    zip_folder_name_only_uuid = zip_file_name.replace(".zip", "")
                    full_path = os.path.join(extract_path, zip_folder_name_only_uuid, order_id)
                    
                    # Define output paths
                    des_path_with_scale = os.path.join(strip_path, zip_folder_name_only_uuid, order_id, "with_scale")
                    des_path_without_scale = os.path.join(strip_path, zip_folder_name_only_uuid, order_id, "without_scale")

                    pkl_path = os.path.join(full_path, "wbc", "Recon_data", "pkl_stacks")

                    # Process and generate strips with scale
                    if process_images_from_pkl(pkl_path, des_path_with_scale, draw_scale=True):
                        print(f"[INFO] Strips with scale created at: {des_path_with_scale}")
                    
                    # Process and generate strips without scale
                    if process_images_from_pkl(pkl_path, des_path_without_scale, draw_scale=False):
                        print(f"[INFO] Strips without scale created at: {des_path_without_scale}")

                    downloaded_files.append(zip_folder_name_only_uuid)
                    
                except Exception as e:
                    print(f"❌ An error occurred while processing {zip_file_name}: {e}")
                
                # Clean up extracted files and downloaded zip
                shutil.rmtree(os.path.join(extract_path, zip_folder_name_only_uuid))
                os.remove(local_zip_path)

                break # Move to the next target file once the corresponding blob is found
        
        if not found_blob:
            print(f"[WARNING] No matching zip file found in GCP for tray_run_id: {tray_run_id}")

    print("\nProcessing complete.")
    print("Matched & downloaded:", downloaded_files)


if __name__ == "__main__":
    main()
