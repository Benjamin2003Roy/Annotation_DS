import pickle

# Path to your pickle file
file_path = "/Users/benjaminroyv/Downloads/b6b42513-4ee7-4d92-a367-422b2cfe9ce4/6d5c201a-ee63-4fa3-a166-5f452adb3206/AnalysisResults/WBC_analyser_result.pkl"

# Load the pickle file
with open(file_path, "rb") as f:
    data = pickle.load(f)

# Print the contents

# If the file contains a large dictionary or nested structure, you can explore keys:
if isinstance(data, dict):
    print(data["6690-2"])
