from dc_inference import *
from tc_inference import *
import os,tqdm,glob
from shutil import copyfile
import time

start = time.time()

input_data_path = "/Users/charani/Desktop/sample_wise_study_patches/HE238_H3"

output_path = "/Users/charani/Desktop/sample_wise_study_patches/HE238_H3_results"

os.makedirs(output_path,exist_ok = True)


tc_model_path = "/Users/charani/Desktop/WBC/DC/codes/codes/WBC_sigvet_inference/models/iter_3_tc.onnx"
dc_model_path = "/Users/charani/Desktop/WBC/DC/codes/codes/WBC_sigvet_inference/models/iter_3_dc.onnx"

tc_class_names = ["debris", "degenerated_cells", "good", "multiple_cells", "out_of_focus"]
dc_class_names =  ["BAND_NEUTROPHILS",'EOSINOPHILS','LYMPHOCYTES','MONOCYTES','NEUTROPHILS','R_N_L_LYMPHOCYTES']




tc_results = os.path.join(output_path,"TC_results")
dc_results = os.path.join(output_path,"DC_results")

os.makedirs(tc_results,exist_ok = True)
os.makedirs(dc_results,exist_ok = True)


if torch.cuda.is_available():
    print('CUDA is available. Working on GPU')
    DEVICE = torch.device('cuda')
else:
    print('CUDA is not available. Working on CPU')
    DEVICE = torch.device('cpu')


wbc_TC_classifier = WBC_TC_Classifier()
wbc_TC_classifier.initialise(tc_model_path,DEVICE)

wbc_DC_classifier = WBC_DC_Classifier()
wbc_DC_classifier.initialise(dc_model_path,DEVICE)

def copy_file_tc(class_name):
    copyfile((os.path.join(input_data_path,img_name)),(os.path.join(tc_results,class_name,img_name)))

def copy_file_dc(class_name):
    copyfile((os.path.join(input_data_path,img_name)),(os.path.join(dc_results,class_name,img_name)))



image_paths = sorted(os.listdir(input_data_path))


img_count = 0
names = []
for img_path in image_paths:
    
    img_name =  os.path.basename(img_path)
    names.append(img_name)
    try:
        if img_name.endswith(".png"):
            img = cv2.imread(os.path.join(input_data_path,img_path))
            output = wbc_TC_classifier.run(img)
            parts = input_data_path.split("/")
            directory_path = "/".join(parts[:-1])
            for class_name in tc_class_names:
                class_path = os.path.join(tc_results,class_name)
                os.makedirs(class_path,exist_ok=True)

            if output == 0:
                copy_file_tc(tc_class_names[0])
            if output == 1:
                copy_file_tc(tc_class_names[1])
            if output == 2:
                copy_file_tc(tc_class_names[2])
            if output == 3:
                copy_file_tc(tc_class_names[3])
            if output == 4:
                copy_file_tc(tc_class_names[4])

            if output == 2:
                output = wbc_DC_classifier.run(img)
                parts = input_data_path.split("/")
                directory_path = "/".join(parts[:-1])
                for class_name in dc_class_names:
                    class_path = os.path.join(dc_results,class_name)
                    os.makedirs(class_path,exist_ok=True)

                if output == 0:
                    copy_file_dc(dc_class_names[0])
                if output == 1:
                    copy_file_dc(dc_class_names[1])
                if output == 2:
                    copy_file_dc(dc_class_names[2])
                if output == 3:
                    copy_file_dc(dc_class_names[3])
                if output == 4:
                    copy_file_dc(dc_class_names[4])
                if output == 5:
                    copy_file_dc(dc_class_names[5])
        else:
            continue
    except Exception as e:
        print(img_path)
        print(e)

names = list(set(names))
print(names)
print(len(names))
end = time.time()

print("total_time: ",end-start)


