import numpy as np
import cv2
import torch
from collections import Counter
from onnx_model_loader import *


model_path = "/Users/charani/Desktop/WBC/DC/codes/codes/WBC_sigvet_inference/models/iter_3_dc.onnx"
class_names = ["EOSINOPHILS","LYMPHOCYTES","MONOCYTES","NEUTROPHILS","R_N_L_LYMPHOCYTES","BAND_NEUTROPHILS"]

class WBC_DC_Classifier():

    def __init__(self) -> None:
        self.model = ONNX_Model_Loader()
        self.model_path = model_path
        self.classes = class_names

    def initialise(self, model_path, device):
        return self.load_model(model_path, device)

    def load_model(self, model_path=None, device='CPU'):
        if model_path is None:
            model_path = self.model_path
        self.model.load_model(model_path, device=device)

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy()

    def pre_process(self, image):
        image = image / 255.0

        # Convert image to torch tensor and reshape
        torch_tensor = torch.from_numpy(np.array(image)).to(torch.float32)

        # If grayscale, uncomment the following line:
        # torch_tensor = torch_tensor.unsqueeze(0) # Add channel dimension for grayscale

        # If RGB, use this:
        torch_tensor = torch_tensor.permute(2, 0, 1)  # Rearrange dimensions (C, H, W)
        torch_tensor = torch_tensor.unsqueeze(0)  # Add batch dimension

        return torch_tensor

    def run(self, image):
        torch_tensor = self.pre_process(image)
        input_names = self.model.get_inputs()
        inputs = {input_names[0]: self.to_numpy(torch_tensor)}
        predictions = self.model.run_inference(inputs)[0]
        classes = np.argmax(predictions, axis=1)
        return classes

    def terminate(self):
        self.model = None
        self.model.unload_model()

    def unload_model(self):
        self.model.unload_model()
