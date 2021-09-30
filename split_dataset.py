import torchvision
import torch
import random


class SplitDS(torch.utils.data.Dataset):
    def __init__(self):
        super(SplitDS, self).__init__()

    def __getitem__(self, index):
        #clrs = [torch.tensor([1, 0, 0], dtype=torch.float),
        #              torch.tensor([0, 1, 0], dtype=torch.float), torch.tensor([0, 0, 1], dtype=torch.float)]
        clrs = [torch.tensor([1, 1, 1], dtype=torch.float),
                      torch.tensor([0, 0, 0], dtype=torch.float)]
        dclr_idx, bclr_idx = random.choice(list(range(len(clrs)))), random.choice(list(range(len(clrs))))
        dclr, bclr = clrs[dclr_idx], clrs[bclr_idx]

        shape = [36, 36, 3]
        img = torch.zeros(shape)
        img[int(shape[0]/2):, :, :] = bclr
        img[:int(shape[0]/2), :, :] = dclr

        return img.transpose(0, 2), (torch.tensor(dclr_idx, dtype=torch.long),
                     torch.tensor(bclr_idx, dtype=torch.long))

    def __len__(self):
        return 10000
