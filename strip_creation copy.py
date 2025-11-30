import os
import pickle
import numpy as np
import cv2
from shutil import copyfile
import time


def process_accession_with_scale(file_path, des_path):

    
    # Create output directory: base_dir/accession_number/with_scale
    output_dir = os.path.join(des_path, "without_scale")
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

    # for x in os.listdir(file_path):
    #     if not os.path.isdir(os.path.join(file_path, x)) or x == ".DS_Store":
    #         continue
    #     if x != accession_number:
    #         continue

    #     for date in os.listdir(os.path.join(file_path, x)):
    #         date_path = os.path.join(file_path, x, date)
    #         if not os.path.isdir(date_path):
    #             continue
                
    #         for tray_id in os.listdir(date_path):
    #             tray_path = os.path.join(date_path, tray_id)
    #             if not os.path.isdir(tray_path):
    #                 continue
                    
    #             for y in os.listdir(tray_path):
    #                 if y != file_name:
    #                     continue

    #                 data_path = os.path.join(tray_path, y)
    pkl_stack_path = os.path.join(file_path, "wbc", "Recon_data", "pkl_stacks")
    # pkl_stack_path= file_path
    
    if not os.path.exists(pkl_stack_path):
        print(f"[WARNING] Path does not exist: {pkl_stack_path}")
        raise FileNotFoundError(f"Path does not exist: {pkl_stack_path}")

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
                    
                    # # Draw scale lines
                    # cv2.line(image, (line_start, y_line), (line_end, y_line), (0, 0, 0), 1)
                    # cv2.line(image, (line_start_h, y_line - 3), (line_start_h, y_line + 3), (0, 0, 0), 1)
                    # cv2.line(image, (line_end_h, y_line - 3), (line_end_h, y_line + 3), (0, 0, 0), 1)
                    # cv2.putText(image, "10", (line_end - 3, y_line + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
                    # cv2.putText(image, "7", (line_end_h - 3, y_line + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
                    # cv2.line(image, (line_start, y_line - 5), (line_start, y_line + 3), (0, 0, 0), 1)
                    # cv2.line(image, (line_end, y_line - 5), (line_end, y_line + 3), (0, 0, 0), 1)

                    # cv2.line(image, (x_line, y_line_start), (x_line, y_line_end), (0, 0, 0), 1)
                    # cv2.line(image, (x_line - 5, y_line_start), (x_line + 5, y_line_start), (0, 0, 0), 1)
                    # cv2.line(image, (x_line - 5, y_line_end), (x_line + 5, y_line_end), (0, 0, 0), 1)

                    # cv2.putText(image, "10", (x_line + 3, y_line_start + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
                    # cv2.putText(image, "7", (x_line + 3, y_line_start_h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
                    # cv2.line(image, (x_line - 3, y_line_start_h), (x_line + 3, y_line_start_h), (0, 0, 0), 1)
                    # cv2.line(image, (x_line - 3, y_line_end_h), (x_line + 3, y_line_end_h), (0, 0, 0), 1)
                    
                    modified_images.append(image)
                else:
                    modified_images.append(padded_image)

            result = np.concatenate(modified_images, axis=1)
            out_name = f"{stack_number}_{data['FOV number']}.png"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, result)
            found_data = True


    output_dir = os.path.join(des_path, "with_scale")
    os.makedirs(output_dir, exist_ok=True)

    pkl_stack_path = os.path.join(file_path, "wbc", "Recon_data", "pkl_stacks")
    # pkl_stack_path= file_path
    
    if not os.path.exists(pkl_stack_path):
        print(f"[WARNING] Path does not exist: {pkl_stack_path}")
        raise FileNotFoundError(f"Path does not exist: {pkl_stack_path}")

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
                    
                    # # Draw scale lines
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

    if found_data:
        print(f"[INFO] Process accession with scale completed: {output_dir}")
        return output_dir
    else:
        print(f"[WARNING] No data found for {accession_number}")
        return None

def main():
    file_path = "/Users/benjaminroyv/Downloads/bdfd2f6c-3e12-4732-a28b-54fe0581b6c4/031b922e-3597-4149-83d8-8c2d50a82cca"
    des_path = file_path
    process_accession_with_scale(file_path, des_path)

if __name__ == "__main__":
    main()