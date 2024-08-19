import json
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image, UnidentifiedImageError
import cv2

class DepthToSketchDataset(Dataset):
    def __init__(self, path_json: str, path_meta: str, resolution: int = 512):
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

        # Load metadata from JSON file
        with open(path_json, 'rt') as f:
            self.data = json.load(f)

        self.list_data = list(self.data.keys())
        self.transform = None  # Optional transform function

    def imread(self, image_path):
        try:
            img = Image.open(image_path)
            if img.mode == 'PA' or img.mode == 'P':
                img = img.convert('RGBA')
            return np.asarray(img.convert('RGB'))
        except UnidentifiedImageError:
            print(f"Cannot identify image file {image_path}")
        except OSError as e:
            print(f"Error processing file {image_path}: {e}")

    def resize_image(self, image, resolution):
        img = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
        return img

    def __len__(self):
        return len(self.data)

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
        filename = self.list_data[idx]

        # Load depth image
        depth_image = self.imread(self.path_meta + "/depth/" + self.data[filename]['depth'])
        if depth_image is None:
            depth_image = torch.zeros((3, self.resolution, self.resolution))  # Handle missing depth image
        else:
            depth_image = self.resize_image(depth_image, self.resolution)
            depth_image = depth_image.astype(np.float32) / 255.0
            depth_image = depth_image.transpose(2, 0, 1)
            depth_image = torch.from_numpy(depth_image)

        # Load sketch image
        sketch_image = self.imread(self.path_meta + "/sketch/" + self.data[filename]['sketch'])
        if sketch_image is None:
            sketch_image = torch.zeros((3, self.resolution, self.resolution))  # Handle missing sketch image
        else:
            sketch_image = self.resize_image(sketch_image, self.resolution)
            sketch_image = sketch_image.astype(np.float32) / 255.0
            sketch_image = sketch_image.transpose(2, 0, 1)
            sketch_image = torch.from_numpy(sketch_image)

        sample = dict(depth_image=depth_image, sketch_image=sketch_image, filename=filename)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
