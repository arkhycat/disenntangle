import torchvision
import torch
import random

#device = "cuda" if torch.cuda.is_available() else "cpu"

def coloring(img):
    clrs = [torch.tensor([1, 0, 0], dtype=torch.float),
                  torch.tensor([0, 1, 0], dtype=torch.float), torch.tensor([0, 0, 1], dtype=torch.float)]
    img = img.expand([3, -1, -1]).transpose(0, 2)
    dclr_idx, bclr_idx = random.sample(list(range(len(clrs))), 2)
    dclr, bclr = clrs[dclr_idx], clrs[bclr_idx]
    img_digit = img * dclr
    img_background = (1-img) * bclr
    return (img_digit+img_background).transpose(0, 2), dclr_idx, bclr_idx

def coloring_frame(img):
    clrs = [torch.tensor([1, 0, 0], dtype=torch.float),
                  torch.tensor([0, 1, 0], dtype=torch.float), torch.tensor([0, 0, 1], dtype=torch.float)]
    img = img.expand([3, -1, -1]).transpose(0, 2)
    dclr_idx, bclr_idx = random.sample(list(range(len(clrs))), 2)
    dclr, bclr = clrs[dclr_idx], clrs[bclr_idx]
    img_digit = img
    new_shape = [int(x) for x in img.shape]
    new_shape[0] += 8
    new_shape[1] += 8
    img_background = torch.zeros(new_shape)
    digit_size = img[:, :, 0].sum()
    img_background[:int(digit_size/8), :, :] = bclr
    #img_background[:, :, :] = bclr
    img_background[4:-4, 4:-4, :] = img
    return img_background.transpose(0, 2), dclr_idx, bclr_idx

class ColMNIST(torchvision.datasets.MNIST):
    def __init__(
            self,
            root,
            train = True,
            transform = None,
            target_transform = None,
            download = True, only_frame = True):
        super(ColMNIST, self).__init__(root, train=train, transform=transform,
                                    target_transform=target_transform, download=download)
        if only_frame:
            self.col_f = coloring_frame
        else:
            self.col_f = coloring

    def __getitem__(self, index):
        img, target = super(ColMNIST, self).__getitem__(index)
        img, dclr_idx, bclr_idx = self.col_f(img)
        return img, (torch.tensor(target),
                     torch.tensor(dclr_idx, dtype=torch.long),
                     torch.tensor(bclr_idx, dtype=torch.long))
