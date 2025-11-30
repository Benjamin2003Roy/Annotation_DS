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
from inference import *
import tqdm,glob
from shutil import copyfile


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

def classes_specified_strips(classified_path, des_path_with_scale, class_strip, classes="LYMPHOCYTES"):
    os.makedirs(class_strip, exist_ok=True)

    class_path = os.path.join(classified_path, classes)
    if not os.path.exists(class_path):
        print(f"[WARNING] Class path does not exist: {class_path}")
        return class_strip

    valid_strip_names = set(os.listdir(des_path_with_scale))

    for image_name in os.listdir(class_path):
        if image_name in valid_strip_names:
            copyfile(
                os.path.join(des_path_with_scale, image_name),
                os.path.join(class_strip, image_name)
            )

    return class_strip

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

def patch_extraction(input_dir, output_dir):
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(input_dir):
        if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue

        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"[WARNING] Could not read image: {image_path}")
            continue

        extracted_patch = extract_input_patch(image)
        cv2.imwrite(os.path.join(output_dir, image_name), extracted_patch)

    return output_dir
    
def classification(input_data_path, output_path):
    start = time.time()
    os.makedirs(output_path, exist_ok=True)

    model_path = "/Users/benjaminroyv/sigvet/strip creation /wbc_tc_dc_iter_14 (1).onnx"

    class_names = [
        'BAND_NEUTROPHILS', 'EOSINOPHILS', 'LYMPHOCYTES', 'MONOCYTES', 'NEUTROPHILS',
        'R_N_L_LYMPHOCYTES', "debris", "degenerated_cells", "degenerated_neutrophils"
    ]

    # GPU / CPU setup
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("CUDA is available. Working on GPU" if DEVICE.type == 'cuda' else "CUDA is not available. Working on CPU")

    # Model
    wbc_classifier = WBC_Classifier()
    wbc_classifier.initialise(model_path, DEVICE)

    # Create class-wise folders
    for cname in class_names:
        os.makedirs(os.path.join(output_path, cname), exist_ok=True)

    # Classification loop
    image_paths = sorted(os.listdir(input_data_path))
    names = set()

    for img_name in image_paths:
        try:
            if not img_name.endswith(".png"):
                continue

            img_path_full = os.path.join(input_data_path, img_name)
            img = cv2.imread(img_path_full)

            output = wbc_classifier.run(img)  # predicted class index
            output = int(np.squeeze(output))
            if 0 <= output < len(class_names):
                target_path = os.path.join(output_path, class_names[output], img_name)
                copyfile(img_path_full, target_path)

            names.add(img_name)
            print(f"Processed: {len(names)} images")

        except Exception as e:
            print("Error:", e)

    print("Total time:", time.time() - start)
    return output_path


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
('563d4103-365a-44c0-be81-1f2bbd15adc9', 'ca7394bc-1e93-4e02-b528-a927b714e8a7'),
('e08ce25f-d58f-457d-b039-7bb98a9f7c06', '921f8591-e87a-4daa-836a-9adb59ac2df1'),
('3147b2dd-b66e-48ef-b167-94f39f9e35f6', '794e2348-df3c-4c23-b221-2c7e26d07750'),
('cb374490-a7bb-41ac-b71e-b17e9a83e435', '87eae45a-a7fc-4a84-b43b-80643d236798'),
('cb374490-a7bb-41ac-b71e-b17e9a83e435', 'd7404f92-9c36-45cd-ad4d-89e16ea521b6'),
('c209f3a0-cb0f-41e7-9a26-b64ae17a9b6e', 'c9a8115b-d0ea-421f-bb21-126f59e0f096'),
('fc010d25-3afa-4f37-b61e-c4172df5c46e', '94b800d0-4efc-4620-a536-df14c974fd45'),
('20111ec5-c933-43f6-87bf-3c7e3b3b8363', 'd220f2f3-9326-4e83-b720-8064b2b8517a'),
('cc18a088-4375-42ce-83bb-089085a2c32e', '17be1c94-eda0-4d31-a215-d52bba0890e9'),
('ea9aafe4-9957-4b76-ae65-cecff843fe0e', '4e40419d-9e35-4ec5-94a3-f493107d58c9'),
('fd4e54b6-3143-46a5-8989-dc6016f1c919', '03fb9e64-06b8-4007-9ae8-f7f0f3c196c6'),
('a79f5178-16b9-48c6-93dd-08d3fa9211e5', '8983ad6c-e3b3-427e-b303-2c3a092f7599'),
('847a8ac2-ad87-4c00-8c43-c512298e6f00', 'c8c82d2b-c8a1-4f4d-9637-8aa4afb4d447'),
('18e8441e-2775-4b1f-a957-8a629f984aad', '783c334e-5ae1-4146-aaa3-4695c22357fa'),
('02a68838-12ed-4ae5-9268-ccb8bedd0544', 'bb64bd2e-3459-451e-8686-a0149540fa77'),
('52e98eb4-6e2d-4ca0-9434-c550f9a79588', 'bb254c8e-1792-4074-a3d3-9c28caaf8580'),
('61668fc9-6e58-4297-b600-cec63c2de47f', '72f3baed-468a-4ad2-a663-f29b124876f0'),
('e2d318c5-3bb1-45e6-b821-d1be0459714a', '0c27c9a3-b872-40a5-929a-8a0dd8ddd1f2'),
('21b1db4d-726e-4989-86e3-29ab4a7f67a4', 'aa2efc32-a8a8-41e1-bc98-f243db1b6cd8'),
('b6981256-4584-456f-aef1-f918eca31496', '673d348d-a8ce-4f2c-ad19-a8763ef234f5'),
('9456d6aa-00b3-432e-9d39-ed5925c626d2', 'e31dcb76-4bed-40d9-acbd-818d744385c5'),
('7223ebb0-eedd-4d26-ad7b-4057f90b6386', '3e91c30f-93b2-4f35-89b1-dd4257900fee'),
('735aa386-8eea-4bcd-92ca-36ac7205ee6b', '818155b5-3373-4c07-a4ad-002407402eae'),
('4d4c0f76-2618-44fb-98fc-355949b41601', 'b8b001ef-1757-4343-a35a-a56d5cbec2cb'),
('35ae9166-a48d-4cb5-a49e-83cd776018bf', 'dc070927-c0d7-4c3d-b3e0-df081ee4e01c'),
('01b6d3bb-f231-4c68-a894-b926acf3fe2b', '2a026156-3bc7-405d-a3bf-052940b562c5'),
('5774478e-93aa-4428-82ba-9817e8f3311b', '6f2729b8-7feb-42c3-b8b1-0480ce8bbbfa'),
('57adfa4a-86a9-4983-ad61-209d4e140a65', '08b0a2a3-9d03-4aa1-9af6-0d6b2d37c5fe'),
('20d7ff9c-bdcf-45af-a999-4e4460330712', 'ee00f925-f3f5-476b-ba07-742cade04247'),
('cf71eba1-89e4-40c4-8a74-d08c60d24099', '6d20b8ff-0e56-4f79-8331-f2a8a4b121bd'),
('514fb2e1-dbf6-43f0-af5f-e16eec8e48e5', 'af812ddf-1b42-42d1-8884-78d1a4e2058f'),
('68e4977b-d04e-472f-87ca-1a46942eacc9', '025c415d-0196-4d07-9570-c4fc75359649'),
('b6683b1e-0483-4d64-b2c1-5207baaeb9ae', '32f7a78d-567b-4c86-9b49-b61af15b3242'),
('376b2ca5-05b7-47d9-ac8a-a47e8e156700', '7b16c94a-9e38-4510-b574-a8e0494f5d82'),
('f28df61f-14fc-4820-b854-446248283a1a', '0e74d8fb-c2d4-4977-b8a8-361cb90fb373'),
('db5ff361-3a41-4793-a8f7-2c01c6c22e8a', '6077eecf-3395-425e-92a4-7b9df4fa7a51'),
('5d32416a-8559-43ab-b750-336aaf575ce4', 'bb3a0971-3c01-42f5-8e29-14806e18c63f'),
('7eb89487-39c0-433f-8606-c54d3603321f', '22a403fa-219b-4125-a26e-65a529e97ac6'),
('4ea06c84-4994-47f9-9468-ee871be89861', '5a310041-3205-4270-b6dd-d0d49e07462b'),
('41f78139-29b1-41ff-9c59-90fbc55f1a0f', '8ecd6e89-300b-4b8c-824a-99c76349d5a7'),
('1e5db55b-9bbb-4815-91ab-1ac9e8b72413', '1c02ac7e-2461-48f5-8d07-d41b8452b49d'),
('f6d3c9b1-c492-4dd4-b2cb-503a7ac18ace', '35a9e467-20a2-42c0-8232-8314ec2bc745'),
('588112e7-fd98-4593-8a1c-97e65746b786', 'da943ace-e1bd-485a-8939-1a5b4e7e7a7a'),
('0ce2c33c-1cc7-4c3a-b2ce-637e6e838859', '4c2a870b-39d0-4da2-8268-3f499e67a8ec'),
('48a9f96d-03d6-420a-bfdd-e31fdf67d0c9', 'a058d360-7d8c-4399-b30e-c3e4eebf1993'),
('6e3dfba4-c366-4d03-8e4d-1bf98f42a859', '6626f4b0-3388-4210-90c9-c0d2a96a4c2b'),
('6e3dfba4-c366-4d03-8e4d-1bf98f42a859', '26ffcdfa-bd99-4842-96d6-b3a4d0538e63'),
('dc98ba03-40cf-40eb-9581-1bb5418f886e', 'cafbf1c6-b4ed-472c-9a76-913b93152a0b'),
('dc98ba03-40cf-40eb-9581-1bb5418f886e', 'ac58e9f6-61ed-4db5-ac92-2219f23ded2f'),
('ad83bcf8-5fc8-4d89-998e-8ddd15c06bb2', '32fed866-3b29-4f15-8ae4-612b630b5d98'),
('e225e76e-9068-4c17-b082-d9f4f70b71be', 'c0a00c15-0eb9-4b57-b45c-2964d0c7a7a2'),
('e225e76e-9068-4c17-b082-d9f4f70b71be', '85cc381b-ce35-478d-bbf8-c72a2ff3999c'),
('29e7345f-897e-4245-b64a-5a74d8e7f7e4', '6eae4df9-a82e-44e3-b451-5a31b01a7265')
]


# ("3f45c785-e717-4b0c-afc5-9716db8498e0"	,"82d4d010-03a6-4941-92d6-5082541beb98")
    
    # --- Setup local directories ---
    os.makedirs(local_path, exist_ok=True)
    extract_path = os.path.join(local_path, "extracted")
    sigvet_mount = mount_smb_share("Sigvet_DC", "/Volumes/Sigvet_DC")
    strip_path = sigvet_mount + "/Ranch_lab_data"
    classes = "LYMPHOCYTES"
    class_strip = os.path.join(strip_path, "class_strips", classes)
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
                    patches_output_dir = os.path.join(strip_path, zip_folder_name_only_uuid, order_id, "patches")
                    classification_output_dir = os.path.join(strip_path, zip_folder_name_only_uuid, order_id, "classification_output")
                    class_strip = os.path.join(strip_path, zip_folder_name_only_uuid, order_id, classes)


                    pkl_path = os.path.join(full_path, "wbc", "Recon_data", "pkl_stacks")

                    # Process and generate strips with scale
                    if process_images_from_pkl(pkl_path, des_path_with_scale, draw_scale=True):
                        print(f"[INFO] Strips with scale created at: {des_path_with_scale}")
                    
                    # Process and generate strips without scale
                    if process_images_from_pkl(pkl_path, des_path_without_scale, draw_scale=False):
                        print(f"[INFO] Strips without scale created at: {des_path_without_scale}")
                    try:
                        print("Starting patch extraction...")
                        patch_extraction(des_path_without_scale,patches_output_dir)
                    except Exception as e:
                        print(f"❌ An error occurred during patch extraction: {e}")
                    try:
                        classified_path=classification(patches_output_dir,classification_output_dir)
                    except Exception as e:
                        print(f"❌ An error occurred during classification: {e}")
                    try:
                        classes_specified_strips(classified_path,des_path_with_scale,class_strip,classes="LYMPHOCYTES")
                    except Exception as e:
                        print(f"❌ An error occurred during class-specific strip extraction: {e}")

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
