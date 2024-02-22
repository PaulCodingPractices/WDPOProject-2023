import torch
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict
import click
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'models/best.pt'
model = torch.load(model_path, map_location=device)['model']
model.to(device).eval()

def iou(box, boxes):
    inter_x1 = np.maximum(boxes[:, 0], box[0])
    inter_y1 = np.maximum(boxes[:, 1], box[1])
    inter_x2 = np.minimum(boxes[:, 2], box[2])
    inter_y2 = np.minimum(boxes[:, 3], box[3])
    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = area_box + area_boxes - inter_area
    union_area = np.maximum(union_area, np.finfo(float).eps)
    return inter_area / union_area


def nms(boxes, scores, iou_threshold):
    indices = np.argsort(scores)[::-1]
    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        if indices.size == 1: break
        ious = iou(boxes[i], boxes[indices[1:]])
        indices = indices[np.where(ious < iou_threshold)[0] + 1]
    return keep

def preprocess_image(img, img_size=640):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).float().unsqueeze(0)
    return img

def apply_nms(predictions, iou_thresh=0.2):
    boxes = predictions[:, :4].cpu().numpy()
    scores = predictions[:, 4].cpu().numpy()
    nms_indices = nms(boxes, scores, iou_thresh)
    return predictions[nms_indices]


def post_process(predictions, obj_thresh=0.3, iou_thresh=0.2):
    filtered_preds = predictions[predictions[:, 4] > obj_thresh]
    nms_preds = apply_nms(filtered_preds, iou_thresh)
    return nms_preds


def draw_final_boxes(img, boxes, scores, class_ids):
    for box, score, class_id in zip(boxes, scores, class_ids):
        x_center, y_center, width, height = box
        x1 = x_center - (width / 2)
        y1 = y_center - (height / 2)
        x2 = x_center + (width / 2)
        y2 = y_center + (height / 2)

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"Class: {class_id}, Score: {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


def detect(img_path: str) -> Dict[str, int]:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_tensor = preprocess_image(img).to(device)
    with torch.no_grad():
        prediction = model(img_tensor)
        predictions = prediction[0][0] if prediction[0].numel() > 0 else torch.tensor([])
        post_processed_preds = post_process(predictions, obj_thresh=0.3, iou_thresh=0.2)

    object_counts = {'aspen': 0, 'birch': 0, 'hazel': 0, 'maple': 0, 'oak': 0}
    if post_processed_preds.numel() > 0: 
        for pred in post_processed_preds:
            scores = pred[5:]
            class_id = scores.argmax().item()
            if class_id == 0:
                object_counts['aspen'] += 1
            elif class_id == 1:
                object_counts['birch'] += 1
            elif class_id == 2:
                object_counts['hazel'] += 1
            elif class_id == 3:
                object_counts['maple'] += 1
            elif class_id == 4:
                object_counts['oak'] += 1
    return object_counts

@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        leaves = detect(str(img_path))
        results[img_path.name] = leaves

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)

if __name__ == '__main__':
    main()