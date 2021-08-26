import torch
from torch.utils.data import Dataset
import numpy as np
import h5py

device = "cuda" if torch.cuda.is_available() else "cpu"

class ThreeDShapes(Dataset):
    def __init__(self, transform = None, filtered = False):
        super(Dataset, self).__init__()
        self.dataset = h5py.File('3dshapes.h5', 'r')
        self.images = self.dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
        self.labels = self.dataset['labels']  # array shape [480000,6], float64
        if filtered:
            l = self.labels
            loc = np.where((l[:, 0] <= 0.1) & 
                            (l[:, 1] <= 0.1) & 
                            (l[:, 2] <= 0.1) & 
                            #(l[:, 3] <= 0.13) & 
                            (l[:, 4] <= 1))# & 
                            #(l[:, 5] <= -25))
            self.images = self.images[loc[0]]
            self.labels = self.labels[loc[0]]

        self.image_shape = self.images.shape[1:]  # [64,64,3]
        self.label_shape = self.labels.shape[1:]  # [6]
        self.n_samples = self.labels.shape[0]  # 10*10*10*8*4*15=480000

        self._FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                             'orientation']
        self._NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                                  'scale': 8, 'shape': 4, 'orientation': 15}
        self.transform=transform
        self.cache_size = 1024
        self.update_cache()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        self.n_cache_accesses += 1
        if self.n_cache_accesses > self.cache_size:
            self.update_cache
        ind = np.random.choice(self.cache_size, 1)[0]
        im = self.cache_imgs[ind]
        im = np.asarray(im)
        im = im / 255. # normalise values to range [0,1]
        im = im.astype(np.float32)
        im = torch.tensor(np.transpose(im.reshape([64, 64, 3]), (2, 0, 1)), device=device)
        labels = self.cache_labels[ind]
        if self.transform:
            im = self.transform(im)
        return im.to(device), labels

    def update_cache(self):
        idxs = np.sort(np.random.choice(self.n_samples, self.cache_size, replace=False)).tolist()
        self.cache_imgs = self.images[idxs]
        self.cache_labels = self.labels[idxs]
        self.n_cache_accesses = 0
