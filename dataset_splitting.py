import os
import random
import shutil
import json

def split_dataset(data_folder, train_ratio=0.95):
    all_files = os.listdir(data_folder)
    image_files = [file for file in all_files if (file.endswith('.jpg') or file.endswith('.png')) and file != 'train.json']

    print(f"Total image files found: {len(image_files)}")

    random.shuffle(image_files)

    num_train = int(len(image_files) * train_ratio)
    train_files = image_files[:num_train]
    val_files = image_files[num_train:]

    print(f"Training files count: {len(train_files)}")
    print(f"Validation files count: {len(val_files)}")

    return train_files, val_files

def create_and_move_files(data_folder, train_files, val_files):
    train_folder = os.path.join(data_folder, 'data_train')
    val_folder = os.path.join(data_folder, 'data_val')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    def move_files(files, destination):
        for file in files:
            shutil.move(os.path.join(data_folder, file), os.path.join(destination, file))
            for ext in ['.xml', '.txt']:
                xml_txt_file = os.path.splitext(file)[0] + ext
                xml_txt_path = os.path.join(data_folder, xml_txt_file)
                if os.path.exists(xml_txt_path):
                    shutil.move(xml_txt_path, os.path.join(destination, xml_txt_file))

    move_files(train_files, train_folder)
    move_files(val_files, val_folder)

def split_json(data_folder, train_files, val_files):
    with open(os.path.join(data_folder, 'train.json'), 'r') as file:
        data = json.load(file)

    train_data = {file: data[file] for file in train_files if file in data}
    val_data = {file: data[file] for file in val_files if file in data}

    with open(os.path.join(data_folder, 'data_train', 'train_data.json'), 'w') as file:
        json.dump(train_data, file, indent=4)

    with open(os.path.join(data_folder, 'data_val', 'val_data.json'), 'w') as file:
        json.dump(val_data, file, indent=4)

data_folder_path = 'data'
train_files, val_files = split_dataset(data_folder_path)
create_and_move_files(data_folder_path, train_files, val_files)
split_json(data_folder_path, train_files, val_files)

print(f"Training files moved: {len(train_files)}")
print(f"Validation files moved: {len(val_files)}")