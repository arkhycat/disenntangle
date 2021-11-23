import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os

#device = "cuda" if torch.cuda.is_available() else "cpu"

class ThreeDShapes(Dataset):
    def __init__(self, filename='3dshapes.h5', transform = None, filtered = False, dt_labels = True, test_dt_labels = False):
        super(Dataset, self).__init__()
        assert(os.path.exists(filename))
        self.dataset = h5py.File(filename, 'r')
        self.images = self.dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
        self.latents = self.dataset['labels']  # array shape [480000,6], float64
        self.dt_labels = dt_labels
        if filtered:
            l = self.latents
            loc = np.where((l[:, 0] <= 0.1) & 
                            (l[:, 1] <= 0.1) & 
                            (l[:, 2] <= 0.1) & 
                            #(l[:, 3] <= 0.13) & 
                            (l[:, 4] <= 1))# & 
                            #(l[:, 5] <= -25))
            self.images = self.images[loc[0]]
            self.latents = self.latents[loc[0]]
        if dt_labels:
            self.n_classes = 8
            l = self.latents

            cond_oh = l[:, 2]>0.5 #object hue cooler
            cond_fh = l[:, 0]>0.5 #floor hue cooler
            cond_or = l[:, 5]>0 #orientation
            cond_shp1 = l[:, 4]>1 #shape
            cond_wh = l[:, 1]>0.5 #wall hue
            cond_scl = l[:, 3]>1 #scale
            cond_shp2 = l[:, 4]>2 #shape

            if not test_dt_labels:
                self.labels = np.zeros((l.shape[0]), dtype=np.int)
                self.labels[cond_oh&cond_fh&cond_or]=7
                self.labels[cond_oh&cond_fh&(~cond_or)]=6
                self.labels[cond_oh&(~cond_fh)&cond_shp1]=5
                self.labels[cond_oh&(~cond_fh)&(~cond_shp1)]=4
                self.labels[(~cond_oh)&cond_wh&cond_scl]=3
                self.labels[(~cond_oh)&cond_wh&(~cond_scl)]=2
                self.labels[(~cond_oh)&(~cond_wh)&cond_shp2]=1
                self.labels[(~cond_oh)&(~cond_wh)&(~cond_shp2)]=0
            else:
                self.labels = np.zeros((l.shape[0]), dtype=np.int)
                self.labels[cond_shp1&cond_shp2&cond_oh]=7
                self.labels[cond_shp1&cond_shp2&(~cond_oh)]=6
                self.labels[cond_shp1&(~cond_shp2)&cond_wh]=5
                self.labels[cond_shp1&(~cond_shp2)&(~cond_wh)]=4
                self.labels[(~cond_shp1)&cond_or&cond_fh]=3
                self.labels[(~cond_shp1)&cond_or&(~cond_fh)]=2
                self.labels[(~cond_shp1)&(~cond_or)&cond_scl]=1
                self.labels[(~cond_shp1)&(~cond_or)&(~cond_scl)]=0

        self.image_shape = self.images.shape[1:]  # [64,64,3]
        self.latent_shape = self.latents.shape[1:]  # [6]
        self.label_shape = self.labels.shape[1:]  # [1]
        self.n_samples = self.latents.shape[0]  # 10*10*10*8*4*15=480000

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
        im = torch.tensor(np.transpose(im.reshape([64, 64, 3]), (2, 0, 1)), dtype=torch.float)
        latents = self.cache_latents[ind]
        if self.transform:
            im = self.transform(im)
        if self.dt_labels:
            labels = self.cached_labels[ind]
            return im, (latents, labels)
        else:
            return im, latents

    def update_cache(self):
        idxs = np.sort(np.random.choice(self.n_samples, self.cache_size, replace=False)).tolist()
        self.cache_imgs = self.images[idxs]
        self.cache_latents = self.latents[idxs]
        if self.dt_labels:
            self.cached_labels = self.labels[idxs]
        self.n_cache_accesses = 0

