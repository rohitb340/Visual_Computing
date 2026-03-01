import os
import os
import numpy as np
from glob import glob
import cv2

def save_npy(data, p):
    dir = os.path.dirname(p)
    os.makedirs(dir, exist_ok=True)
    np.save(p, data)


save_dir = ''  # PATH TO FUNDUS DATA HERE
orig_dir = '' # PATH TO FUNDUS DATA HERE

os.makedirs(save_dir, exist_ok=True)

for site in ['Drishti-GS', 'RIM-ONE', 'REFUGE_t', 'REFUGE_v']:

    image_paths = glob(f'{orig_dir}/{site}/*/image/*.png')

    for image_path in image_paths:
        mask_path = image_path.replace('image', 'mask')
        assert (os.path.exists(mask_path))
        img = cv2.imread(image_path)[:, :, ::-1]
        img = cv2.resize(img, (384, 384), cv2.INTER_CUBIC)

        mask = 2 - np.array(cv2.imread(mask_path, 0) / 127, dtype='uint8')
        mask = cv2.resize(mask, (384, 384), cv2.INTER_NEAREST)
        train_or_test = image_path.split('/')[-3]
        file_name = image_path.split('/')[-1][:-4]

        save_npy(img, f'{save_dir}/{site}/{train_or_test}/image/{file_name}')
        save_npy(mask, f'{save_dir}/{site}/{train_or_test}/mask/{file_name}')