import onnxruntime

class ONNX_Model_Loader():

    def __init__(self):
        self.model_path = None
        self.session = None

    def load_model(self, model_path, device):

        self.model_path = model_path
        print(device)
       
        if device == "CPU":
            execution_providers = ["CPUExecutionProvider"]
        else:
            execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = onnxruntime.InferenceSession(
        self.model_path, providers=execution_providers
        )

    def unload_model(self):

        if self.session:
            self.session = None
            print("ONNX model unloaded successfully.")
        else:
            print("No ONNX model is loaded.")

    def get_model_input_output_names(self):
        
        if not self.session:
            raise ValueError("No ONNX model is loaded.")
        
        input_names = [inp.name for inp in self.session.get_inputs()]
        output_names = [out.name for out in self.session.get_outputs()]

        return input_names, output_names
    
    def get_inputs(self):
        return [input.name for input in self.session.get_inputs()]

    def run_inference(self, input_data):
       
        if not self.session:
            raise ValueError("No ONNX model is loaded.")
        
        input_names, output_names = self.get_model_input_output_names()
        results = self.session.run(output_names, input_data)
        return results





