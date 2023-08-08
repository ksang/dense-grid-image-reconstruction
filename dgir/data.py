import os
import numpy as np
import torch
import torchvision.io as io
from tqdm import tqdm
from torch.utils.data import Dataset

class DenseGridImageDataset(Dataset):
    """Dense grid image dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        # raw image data
        self.images = {}
        self.load_images()
        # generate samples [id, u, v, r, g, b]
        self.generate_samples()

    def load_images(self):
        files = []
        for images in os.listdir(self.root_dir):
            if (images.endswith(".png") or images.endswith(".jpg")):
                files.append(os.path.join(self.root_dir, images))

        for i, f in enumerate(tqdm(files)):
            d = io.read_image(f)[:3,:,:]
            self.images[i+1] = {
                "file": f,
                "data": d,
                "shape": d.shape,
            }

    def generate_samples(self):
        """
        Sample format:
            [id, u, v, r, g, b]
        """
        sample_list = []
        for image_id in tqdm(self.images.keys()):
            im = self.images[image_id]
            # image is stores in (C, H ,W) format
            grid_x = torch.linspace(0, 1, im["shape"][2])
            grid_y = torch.linspace(0, 1, im["shape"][1])
            axis_y, axis_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
            uv = torch.stack([axis_x, axis_y])
            samples = torch.cat([uv, im["data"]]).view(5,-1)    # [u, v, r, g, b]
            samples = torch.cat([torch.ones(1, samples.size(1))*image_id, samples])
            sample_list.append(samples.permute(1, 0))
        self.samples = torch.cat(sample_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample