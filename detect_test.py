import torch
import cv2
import numpy as np


def preprocess_image(img_path, img_size=640, maintain_aspect_ratio=True):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if maintain_aspect_ratio:
        old_size = img.shape[:2]
        ratio = float(img_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = img_size - new_size[1]
        delta_h = img_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        pad = (left, top, right, bottom)

        padded_height = new_size[0] + top + bottom
        padded_width = new_size[1] + left + right
        if padded_width <= 0 or padded_height <= 0:
            raise ValueError("Padding has resulted in non-positive dimensions of the image.")

        current_size = (padded_width, padded_height)
        print(f"After padding, current size: {current_size}")
    else:
        img = cv2.resize(img, (img_size, img_size))
        current_size = (img_size, img_size)
        pad = (0, 0, 0, 0)

    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.tensor(img, dtype=torch.float).unsqueeze(0)
    return img_tensor, current_size, pad

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
        print(f"Indices before filtering: {indices.size}")
        indices = indices[np.where(ious < iou_threshold)[0] + 1]
        print(f"Indices after filtering: {indices.size}")
    return keep


def apply_nms(predictions, iou_thresh=0.8):
    print("apply_nms called")
    boxes = predictions[:, :4].cpu().numpy()
    scores = predictions[:, 4].cpu().numpy()
    print(f"Number of predictions before NMS: {len(boxes)}")
    nms_indices = nms(boxes, scores, iou_thresh)
    print(f"Number of predictions after NMS: {len(nms_indices)}")
    return predictions[nms_indices]

def post_process(predictions, obj_thresh=0.5, iou_thresh=0.8):
    print(f"Predictions before objectness filtering: {predictions}")
    filtered_preds = predictions[predictions[:, 4] > obj_thresh]
    print(f"Predictions after objectness filtering: {filtered_preds}")
    nms_preds = apply_nms(filtered_preds, iou_thresh)
    return nms_preds


def adjust_boxes(boxes, orig_size, current_size, pad=(0, 0, 0, 0)):
    orig_width, orig_height = orig_size
    current_width, current_height = current_size
    pad_left, pad_top, pad_right, pad_bottom = pad

    if current_width - pad_left - pad_right <= 0:
        raise ValueError(f"Invalid current width for scaling: {current_width} with padding: {pad_left}, {pad_right}")
    if current_height - pad_top - pad_bottom <= 0:
        raise ValueError(f"Invalid current height for scaling: {current_height} with padding: {pad_top}, {pad_bottom}")

    x_scale = orig_width / (current_width - pad_left - pad_right)
    y_scale = orig_height / (current_height - pad_top - pad_bottom)

    # Debug information
    print(f"Original Size: {orig_size}, Current Size: {current_size}, Padding: {pad}")
    print(f"Scaling factors - X: {x_scale}, Y: {y_scale}")

    adjusted_boxes = []
    for index, box in enumerate(boxes):
        x_center, y_center, width, height = box
        x1 = (x_center - width / 2 - pad_left) * x_scale
        y1 = (y_center - height / 2 - pad_top) * y_scale
        x2 = (x_center + width / 2 - pad_left) * x_scale
        y2 = (y_center + height / 2 - pad_top) * y_scale

        x1 = np.clip(x1, 0, orig_width)
        y1 = np.clip(y1, 0, orig_height)
        x2 = np.clip(x2, 0, orig_width)
        y2 = np.clip(y2, 0, orig_height)

        adjusted_box = [x1, y1, x2, y2]
        adjusted_boxes.append(adjusted_box)
        print(f"Box {index + 1} - Original: {box}, Adjusted: {adjusted_box}")

    return np.array(adjusted_boxes)

def draw_final_boxes(img, boxes, scores, class_ids, orig_size, current_size, pad=(0, 0, 0, 0)):
    adjusted_boxes = adjust_boxes(boxes, orig_size, current_size, pad)
    for box, score, class_id in zip(adjusted_boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"Class: {class_id}, Score: {score:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

def enhanced_detect(img_path: str, model, device, obj_thresh=0.4, iou_thresh=0.7, class_thresh=0.4):
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    original_height, original_width = original_img.shape[:2]

    img_tensor, current_size, pad = preprocess_image(img_path, maintain_aspect_ratio=True)
    img_tensor = img_tensor.to(device)

    model.eval()
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

    for pred in post_processed_preds:
        bbox = pred[:4].cpu().numpy()
        obj_score = pred[4].item()
        class_scores = pred[5:].cpu().numpy()
        best_class_idx = np.argmax(class_scores)
        best_class_score = class_scores[best_class_idx]


        print(f"Bounding Box: {bbox}, Objectness: {obj_score:.2f}, Best Class: {best_class_idx} (Score: {best_class_score:.2f})")

    class_scores = post_processed_preds[:, 5:]
    scores, class_ids = class_scores.max(1)

    print(f"Class scores: {class_scores}")
    print(f"Final scores: {scores}")
    print(f"Final class IDs: {class_ids}")

    mask = scores > class_thresh
    final_boxes = post_processed_preds[mask][:, :4]
    final_scores = scores[mask]
    final_class_ids = class_ids[mask]

    print(f"Final boxes: {final_boxes}")
    print(f"Final scores: {final_scores}")
    print(f"Final class IDs: {final_class_ids}")

    img = cv2.imread(img_path)

    orig_size = (original_width, original_height)


    original_img = cv2.imread(img_path)


    print(f"Drawing boxes with orig_size: {orig_size}, current_size: {current_size}, pad: {pad}")

    draw_final_boxes(original_img, final_boxes, final_scores, final_class_ids,
                     (original_width, original_height), current_size, pad)

    output_path = img_path.replace(".jpg", "_debug.jpg")
    cv2.imwrite(output_path, original_img)
    print(f"Output saved to {output_path}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'models/best.pt'
model = torch.load(model_path, map_location=device)['model']
model.to(device).eval()

img_path = 'data/data_val/images/0031.jpg'
enhanced_detect(img_path, model, device)