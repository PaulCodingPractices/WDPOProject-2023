
import torch

# Load the model checkpoint
loaded_checkpoint = torch.load('C:\\WDPOProject-2023\\models\\best.pt', map_location=torch.device('cpu'))

# Assuming 'model' key in checkpoint contains the entire model
model = loaded_checkpoint['model']

# If the 'model' key is actually a state dict, you'll need to instantiate the model architecture first
# from models.yolo import Model  # Ensure correct import based on your setup
# model_cfg_path = 'C:\\yolov5-master\\models\\yolov5m.yaml'  # Path to model config
# model = Model(cfg=model_cfg_path, ch=3, nc=5)  # Instantiate model
# model.load_state_dict(loaded_checkpoint['model'])

model.eval()  # Set the model to evaluation mode
print("Model loaded successfully.")

