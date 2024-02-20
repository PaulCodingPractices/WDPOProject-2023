import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm
import torch
import numpy as np


def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        A dictionary with the number of each object.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    def preprocess_image(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img / 255.0
        img = img.transpose((2, 0, 1))
        return img

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

    def apply_nms(predictions, iou_thresh=0.2):
        boxes = predictions[:, :4].cpu().numpy()
        scores = predictions[:, 4].cpu().numpy()
        nms_indices = nms(boxes, scores, iou_thresh)
        return predictions[nms_indices]

    def post_process(predictions, obj_thresh=0.3, iou_thresh=0.2):
        filtered_preds = predictions[predictions[:, 4] > obj_thresh]
        nms_preds = apply_nms(filtered_preds, iou_thresh)
        return nms_preds

    def enhanced_detect(model, device, obj_thresh=0.3, iou_thresh=0.2, class_thresh=0.3):
        img_tensor = preprocess_image(img).to(device)

        if device.type == 'cuda':
            model.half()
            img_tensor = img_tensor.half()
        else:
            model.float()
            img_tensor = img_tensor.float()

        with torch.no_grad():
            prediction = model(img_tensor)

        predictions = prediction[0][0] if prediction[0].numel() > 0 else torch.tensor([])

        post_processed_preds = post_process(predictions, obj_thresh, iou_thresh)

        class_scores = post_processed_preds[:, 5:]
        scores, class_ids = class_scores.max(1)

        mask = scores > class_thresh
        final_boxes = post_processed_preds[mask][:, :4]
        final_scores = scores[mask]
        final_class_ids = class_ids[mask]

        return {'boxes': final_boxes, 'scores': final_scores, 'class_ids': final_class_ids}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'models/best.pt'
    model = torch.load(model_path, map_location=device)['model']
    model.to(device).eval()

    detection_results = enhanced_detect(model, device)

    aspen = birch = hazel = maple = oak = 0
    for class_id in detection_results['class_ids']:
        if class_id == 0:
            aspen += 1
        elif class_id == 1:
            birch += 1
        elif class_id == 2:
            hazel += 1
        elif class_id == 3:
            maple += 1
        elif class_id == 4:
            oak += 1

    return {'aspen': aspen, 'birch': birch, 'hazel': hazel, 'maple': maple, 'oak': oak}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory',
              type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
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