from inference import *

import os,tqdm,glob
from shutil import copyfile
import time

def calssification(input_data_path,output_path)
    start = time.time()

    input_data_path = "/Users/charani/Desktop/wbc_tc_dc_data/iter_4/data/kmeans/neut_like_2/dbscan_cluster_14"

    output_path = "/Users/charani/Desktop/wbc_tc_dc_data/iter_4/data/kmeans/neut_like_2/results/dbscan_cluster_14_Results"
    os.makedirs(output_path,exist_ok = True)

    model_path = "/Users/charani/Desktop/sigvet_models/wbc_tc_dc_iter_1.onnx"

    #class_names =  ['BAND_NEUTROPHILS','EOSINOPHILS','LYMPHOCYTES','MONOCYTES','NEUTROPHILS','R_N_L_LYMPHOCYTES',"NEUTROPHILS_LIKE_2","debris","degenerated_cells","multiple_cells","out_of_focus","degenerated_neutrophils"]
    class_names =  ['BAND_NEUTROPHILS','EOSINOPHILS','LYMPHOCYTES','MONOCYTES','NEUTROPHILS','R_N_L_LYMPHOCYTES',"debris","degenerated_cells","multiple_cells","out_of_focus","degenerated_neutrophils"]

    os.makedirs(output_path,exist_ok = True)
    csv_path = os.path.join(output_path,"prob_csv.csv")



    if torch.cuda.is_available():
        print('CUDA is available. Working on GPU')
        DEVICE = torch.device('cuda')
    else:
        print('CUDA is not available. Working on CPU')
        DEVICE = torch.device('cpu')



    wbc_classifier = WBC_Classifier()
    wbc_classifier.initialise(model_path,DEVICE)

    def copy_file(class_name):
        copyfile((os.path.join(input_data_path,img_name)),(os.path.join(output_path,class_name,img_name)))


    for class_name in class_names:
        class_path = os.path.join(output_path,class_name)
        os.makedirs(class_path,exist_ok=True)

    image_paths = sorted(os.listdir(input_data_path))


    img_count = 0
    names = []
    for img_path in image_paths:
        
        img_name =  os.path.basename(img_path)
        names.append(img_name)
        try:
            if img_name.endswith(".png"):
                img = cv2.imread(os.path.join(input_data_path,img_path))
                output,prob = wbc_classifier.run(img,csv_path)
            

                if output == 0:
                    copy_file(class_names[0])
                if output == 1:
                    copy_file(class_names[1])
                if output == 2:
                    copy_file(class_names[2])
                if output == 3:
                    copy_file(class_names[3])
                if output == 4:
                    copy_file(class_names[4])
                if output == 5:
                    copy_file(class_names[5])
                if output == 6:
                    copy_file(class_names[6])
                if output == 7:
                    copy_file(class_names[7])
                if output == 8:
                    copy_file(class_names[8])
                if output == 9:
                    copy_file(class_names[9])
                if output == 10:
                    copy_file(class_names[10])
                if output == 11:
                    copy_file(class_names[11])
                

                
            else:
                continue
        except Exception as e:
        
            print(e)

    names = list(set(names))
    print(len(names))
    end = time.time()

    print("total_time: ",end-start)


