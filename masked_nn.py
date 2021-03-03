import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.mask = torch.ones(10, 50)
        self.original_w = None

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        self.last_layer = F.relu(self.fc1(x))
        x = F.dropout(self.last_layer, training=self.training)
        self.fc2.weight = nn.Parameter(self.fc2.weight * self.mask)
        x = self.fc2(x)
        return F.log_softmax(x)

    def set_mask(self, mask):
        if self.original_w is None:
            self.original_w = self.fc2.weight
        else:
            self.fc2.weight = self.original_w
        self.mask = mask
