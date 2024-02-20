import cv2
import torch
import numpy as np


def preprocess_image(img_path, img_size=640):
    """Preprocess the input image."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).float()  # Use float here for compatibility
    img = img.unsqueeze(0)  # Add batch dimension
    return img

def test_inference(img_path, model, device):
    img_tensor = preprocess_image(img_path).to(device)

    if device.type == 'cuda':
        model.half()  # Use half precision if on GPU
        img_tensor = img_tensor.half()
    else:
        model.float()  # Ensure full precision on CPU
        img_tensor = img_tensor.float()

    with torch.no_grad():
        output = model(img_tensor)

    print("Output Type:", type(output))
    print("Output Structure:", output)

    if isinstance(output, (list, tuple)):
        for i, item in enumerate(output):
            print(f"Element {i}: Type={type(item)}, Shape={item.shape if torch.is_tensor(item) else 'N/A'}")

    elif torch.is_tensor(output):
        print("Output Shape:", output.shape)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'models/best.pt'  # Update this path
model = torch.load(model_path, map_location=device)['model']
model.to(device).eval()

img_path = 'data/data_val/images/0035.jpg'

test_inference(img_path, model, device)