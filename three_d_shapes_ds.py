import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
from joblib import load

#device = "cuda" if torch.cuda.is_available() else "cpu"

class ThreeDShapes(Dataset):
    def __init__(self, filename='3dshapes.h5', transform = None, train=True, filtered = False, dt_labels = True, test_dt_labels = False, test_split=4):
        super(Dataset, self).__init__()
        assert(os.path.exists(filename))
        self.dataset = h5py.File(filename, 'r')
        self.images = np.array(self.dataset['images'][:])  # array shape [480000,64,64,3], uint8 in range(256)
        self.latents = np.array(self.dataset['labels'][:])  # array shape [480000,6], float64
        print(type(self.images))
        self.dt_labels = dt_labels
        if filtered:
            l = self.latents
            self.images = self.images[loc[0]]
            self.latents = self.latents[loc[0]]
        if dt_labels:
            self.n_classes = 8
            l = self.latents

            if not test_dt_labels:
                dt = load('decision_tree_train.joblib') 
                self.labels = dt.predict(l)
            else:
                dt = load('decision_tree_test.joblib') 
                self.labels = dt.predict(l)

        self.image_shape = self.images.shape[1:]  # [64,64,3]
        self.latent_shape = self.latents.shape[1:]  # [6]
        self.label_shape = self.labels.shape[1:]  # [1]
        self.full_size = self.latents.shape[0] #10*10*10*8*4*15=480000
        if train:
            self.n_samples = int(self.latents.shape[0]/test_split*(test_split-1)) 
        else:
            self.n_samples = int(self.latents.shape[0]/test_split)


        self._FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                             'orientation']
        self._NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                                  'scale': 8, 'shape': 4, 'orientation': 15}
        self.transform=transform
        self.load_and_shuffle(train, test_split)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        im = self.cache_imgs[idx]
        im = np.asarray(im)
        im = im / 255. # normalise values to range [0,1]
        im = im.astype(np.float32)
        im = torch.tensor(np.transpose(im.reshape([64, 64, 3]), (2, 0, 1)), dtype=torch.float)
        latents = self.cache_latents[idx]
        if self.transform:
            im = self.transform(im)
        if self.dt_labels:
            labels = self.cached_labels[idx]
            return im, (latents, labels)
        else:
            return im, latents

    def load_and_shuffle(self, train, test_split):
        idxs = np.arange(self.full_size) #np.random.choice(self.full_size, self.full_size, replace=False)
        if train:
            ii = np.where(idxs % test_split != 0)[0]
            idxs = idxs[ii]
        else:
            ii = np.where(idxs % test_split == 0)[0]
            idxs = idxs[ii]
        idxs = idxs.tolist()
        self.cache_imgs = self.images[idxs]
        self.cache_latents = self.latents[idxs]
        if self.dt_labels:
            self.cached_labels = self.labels[idxs]
        print("done loading")

