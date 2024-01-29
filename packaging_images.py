import os
import shutil

base_dir = 'data'

sub_dirs = ['data_train', 'data_val']

for sub_dir in sub_dirs:
    images_dir = os.path.join(base_dir, sub_dir, 'images')
    labels_dir = os.path.join(base_dir, sub_dir, 'labels')

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    for item in os.listdir(os.path.join(base_dir, sub_dir)):
        if item.endswith('.jpg'):
            shutil.move(os.path.join(base_dir, sub_dir, item), images_dir)

print("images where moved to theyre respective packages")