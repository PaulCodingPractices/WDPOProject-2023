import torch
import cv2
import numpy as np


def preprocess_image(img_path, img_size=640):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).float().unsqueeze(0)
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
        # Diagnostic print to check the effect of IoU thresholding
        print(f"Indices before filtering: {indices.size}")
        indices = indices[np.where(ious < iou_threshold)[0] + 1]
        print(f"Indices after filtering: {indices.size}")
    return keep


def apply_nms(predictions, iou_thresh=0.2):
    print("apply_nms called")
    boxes = predictions[:, :4].cpu().numpy()
    scores = predictions[:, 4].cpu().numpy()
    print(f"Number of predictions before NMS: {len(boxes)}")
    nms_indices = nms(boxes, scores, iou_thresh)
    print(f"Number of predictions after NMS: {len(nms_indices)}")
    return predictions[nms_indices]


def post_process(predictions, obj_thresh=0.3, iou_thresh=0.2):
    print(f"Predictions before objectness filtering: {predictions}")
    filtered_preds = predictions[predictions[:, 4] > obj_thresh]
    print(f"Predictions after objectness filtering: {filtered_preds}")
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


def enhanced_detect(img_path: str, model, device, obj_thresh=0.3, iou_thresh=0.2, class_thresh=0.3):
    img_tensor = preprocess_image(img_path).to(device)

    if device.type == 'cuda':
        model.half()
        img_tensor = img_tensor.half()
    else:
        model.float()
        img_tensor = img_tensor.float()

    with torch.no_grad():
        prediction = model(img_tensor)

    print(f"Raw predictions: {prediction}")

    predictions = prediction[0][0] if prediction[0].numel() > 0 else torch.tensor([])
    print(f"Predictions after accessing the expected tensor: {predictions}")

    post_processed_preds = post_process(predictions, obj_thresh, iou_thresh)

    print(f"Predictions after NMS: {post_processed_preds}")

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

    draw_final_boxes(img, final_boxes.cpu().numpy(), final_scores.cpu().numpy(), final_class_ids.cpu().numpy())


    output_path = img_path.replace(".jpg", "_debug.jpg")
    cv2.imwrite(output_path, img)
    print(f"Output saved to {output_path}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'models/best.pt'
model = torch.load(model_path, map_location=device)['model']
model.to(device).eval()

img_path = 'data/data_val/images/0035.jpg'
enhanced_detect(img_path, model, device)