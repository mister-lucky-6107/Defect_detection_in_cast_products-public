#import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch

from set_seed import set_seed
set_seed(42)


def load_and_preprocess_images(df, transform, train_data_dir, masked_dir, seg_masked_dir, num_augmentations=1):
    images = []
    labels = []
    seg_masks = []
    image_paths = []

    for _, row in df.iterrows():
        orig_name = row['id']
        base_name = orig_name.replace('.jpeg', '')
        #orig_path = os.path.join(train_data_dir, orig_name)
        #mask_path = os.path.join(masked_dir, base_name + '_masked.png')
        #seg_mask_path = os.path.join(seg_masked_dir, base_name + '_mask.png')
        orig_path = Path(train_data_dir) / orig_name
        mask_path = Path(masked_dir) / (base_name + '_masked.png')
        seg_mask_path = Path(seg_masked_dir) / (base_name + '_mask.png')

        for _ in range(num_augmentations):

            # 画像読み込み
            #if os.path.exists(mask_path):
            if mask_path.exists():
                image = Image.open(mask_path).convert('RGB')
            else:
                image = Image.open(orig_path).convert('RGB')
            image = np.array(image)

            # マスク読み込み
            #if os.path.exists(seg_mask_path):
            if seg_mask_path.exists():
                seg_mask = Image.open(seg_mask_path).convert('L')
                seg_mask = np.array(seg_mask)
            else:
                seg_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            # 画像とマスクを同時にtransform
            augmented = transform(image=image, mask=seg_mask)
            images.append(augmented['image'])
            labels.append(torch.tensor(row['target']))
            seg_masks.append(torch.tensor(augmented['mask'], dtype=torch.long))
            #image_paths.append(mask_path if os.path.exists(mask_path) else orig_path)
            image_paths.append(mask_path if mask_path.exists() else orig_path)

    all_images = torch.stack(images)
    all_labels = torch.tensor(labels)
    all_seg_masks = torch.stack(seg_masks)
    all_image_paths = np.array(image_paths)

    return all_images, all_labels, all_seg_masks, all_image_paths