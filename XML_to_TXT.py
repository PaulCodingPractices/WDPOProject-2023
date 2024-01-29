import os
import xml.etree.ElementTree as ET

def convert_annotation(xml_file, output_txt_file, class_mapping):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_size = root.find('size')
    w = int(image_size.find('width').text)
    h = int(image_size.find('height').text)

    with open(output_txt_file, 'w') as output_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in class_mapping:
                continue
            cls_id = class_mapping[cls]
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = (b[0] / w, b[2] / h, b[1] / w, b[3] / h)  # YOLO format
            output_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

class_mapping = {'aspen': 0, 'birch': 1, 'hazel': 2, 'maple': 3, 'oak': 4}

data_train_dir = 'data/data_train'
data_val_dir = 'data/data_val'
xml_dir = 'boxed_data'

os.makedirs(os.path.join(data_train_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(data_val_dir, 'labels'), exist_ok=True)

for xml_file in os.listdir(xml_dir):
    if xml_file.endswith('.xml'):
        base_filename = os.path.splitext(xml_file)[0]
        if os.path.exists(os.path.join(data_train_dir, base_filename + '.jpg')):
            output_txt_file = os.path.join(data_train_dir, 'labels', base_filename + '.txt')
            convert_annotation(os.path.join(xml_dir, xml_file), output_txt_file, class_mapping)
            print(f"Processed {xml_file} into {output_txt_file}")
        elif os.path.exists(os.path.join(data_val_dir, base_filename + '.jpg')):
            output_txt_file = os.path.join(data_val_dir, 'labels', base_filename + '.txt')
            convert_annotation(os.path.join(xml_dir, xml_file), output_txt_file, class_mapping)
            print(f"Processed {xml_file} into {output_txt_file}")
        else:
            print(f"No corresponding image found for {xml_file} in data_train or data_val")

print("Conversion and organization of annotations completed.")