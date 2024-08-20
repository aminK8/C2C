import json
import numpy as np
import torch
import cv2
from torch.utils.data.dataset import Dataset
from PIL import Image, UnidentifiedImageError
from typing import Dict
import random


class DepthToSketchDataset(Dataset):
    def __init__(self, path_json_depth: str, path_json_sketch: str, path_meta: str, resolution: int = 256):
        """
        Depth to Sketch Dataset/Dataloader.

        :param path_json: (formatted) path to MultiGen20M json files,
            e.g. "path/to/json_files/depth_sketch_group_{}.json"
        :param path_meta: path to the data root folder
        :param resolution: target resolution for images
        """
        super().__init__()
        self.path_meta = path_meta
        self.resolution = resolution
        self.data_depth = []
        self.data_sketch = []

        with open(path_json_depth, 'rt') as f:
            for line in f:
                self.data_depth.append(json.loads(line))

        with open(path_json_sketch, 'rt') as f:
            for line in f:
                self.data_sketch.append(json.loads(line))

        # self.list_data = list(self.data.keys())
        self.transform = None  # Optional transform function

    def imread(self, image_path, mode='RGB'):
        try:
            img = Image.open(image_path)
            if img.mode == 'PA' or img.mode == 'P':
                img = img.convert('RGBA')
            return np.asarray(img.convert(mode))
        except UnidentifiedImageError:
            print(f"Cannot identify image file {image_path}")
        except OSError as e:
            print(f"Error processing file {image_path}: {e}")

    def __len__(self):
        return len(self.data_sketch)

    def __getitem__(self, idx):
        """
        The __getitem__ method to iterate over the dataset.

        Parameters
        ----------
        idx : int
            index of the current file in the dataset.

        Returns
        -------
        dict
            "depth_img" : depth image numpy array (C, H, W)
            "sketch_img" : sketch image numpy array (C, H, W)
        """
        # filename = self.list_data[idx]

        # Load depth image
        depth_image = self.imread(self.path_meta + self.data_depth[idx]['control_depth'].replace("aesthetics_6_25_plus_", ""))
        if depth_image is None:
            depth_image = torch.zeros((3, self.resolution, self.resolution))  # Handle missing depth image
        else:
            depth_image = depth_image.astype(np.float32) / 255.0

        # Load sketch image
        sketch_image = self.imread(self.path_meta + self.data_sketch[idx]['control_hed'].replace("aesthetics_6_25_plus_", ""), mode='L')
        if sketch_image is None:
            sketch_image = torch.zeros((1, self.resolution, self.resolution))  # Handle missing sketch image

        sample = dict(depth_image=depth_image,
                      sketch_image=sketch_image,
                      filename_sketch=self.data_sketch[idx]['control_hed'].replace("aesthetics_6_25_plus_", ""),
                      filename_depth= self.data_depth[idx]['control_depth'].replace("aesthetics_6_25_plus_", ""))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def resize_all(sample, tgt_size):
    h, w, c = sample['depth_image'].shape
    sample['depth_image'] = cv2.resize(sample['depth_image'], tgt_size, interpolation=cv2.INTER_LINEAR)
    sample['sketch_image'] = cv2.resize(sample['sketch_image'], tgt_size, interpolation=cv2.INTER_LINEAR)
    return sample


def normalize(img, mean, std):
    assert img.shape[2] == 3
    return (img - mean[None, None, :]) / std[None, None, :]


def horizontal_flip_all(sample):
    sample['depth_image'] = sample['depth_image'][:, ::-1, :].copy()  # (H, W, C)
    sample['sketch_image'] = sample['sketch_image'][:, ::-1].copy()  # (H, W)
    return sample


class CondCvtrTransform:
    def __init__(self,
                 cfg: Dict):
        # resize
        self.image_size = cfg['resize'] if cfg['do_resize'] else None
        # normalize
        self.image_mean = np.array(cfg['image_mean']) if cfg['do_normalize'] else None
        self.image_std = np.array(cfg['image_std']) if cfg['do_normalize'] else None
        assert self.image_mean.shape[0] == 3
        assert self.image_std.shape[0] == 3
        # horizontal flip
        self.flip_p = cfg['flip_p'] if cfg['do_flip'] else None

    def __call__(self, sample):
        # resize
        if self.image_size is not None:
            sample = resize_all(sample, tgt_size=self.image_size)
        # normalize
        if self.image_mean is not None:
            sample['depth_image'] = normalize(sample['depth_image'], mean=self.image_mean, std=self.image_std)
        # flip
        if self.flip_p is not None and random.random() > self.flip_p:
            sample = horizontal_flip_all(sample)
        # to Tensor
        sample['depth_image'] = torch.tensor(sample['depth_image'].transpose(2, 0, 1)).to(torch.float32)  # (C, H, W)
        sample['sketch_image'] = torch.from_numpy(sample['sketch_image']).to(torch.long)  # (H, W)
        return sample
