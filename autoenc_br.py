from col_mnist import ColMNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from spectral_utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"


class AE(nn.Module):
    def __init__(self, input_shape, bottleneck_size=100):
        super().__init__()
        self.hidden_size = 128
        self.original_w = None

        self.encoder_hidden_layer = nn.Linear(
            in_features=input_shape, out_features=self.hidden_size
        )
        self.encoder_output_layer = nn.Linear(
            in_features=self.hidden_size, out_features=bottleneck_size
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=bottleneck_size, out_features=self.hidden_size
        )
        self.decoder_output_layer = nn.Linear(
            in_features=self.hidden_size, out_features=input_shape
        )

        self.mask = None

    def forward(self, features):
        if self.mask is not None:
            self.encoder_output_layer.weight = nn.Parameter(self.encoder_output_layer.weight.to(device) * self.mask.to(device))
            self.decoder_hidden_layer.weight = nn.Parameter(self.decoder_hidden_layer.weight.to(device) * torch.transpose(self.mask, 1, 0).to(device))
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

    def set_mask(self, mask):
        if self.original_w is None:
            self.original_w = [self.encoder_output_layer.weight.clone(), self.decoder_hidden_layer.weight.clone()]
        else:
            self.encoder_output_layer.weight = nn.Parameter(self.original_w[0].clone())
            self.decoder_hidden_layer.weight = nn.Parameter(self.original_w[1].clone())
        self.mask = mask

class Trainer:
    def __init__(self, n_conn_comp, batch_size_train = 32, batch_size_test= 32):
        self.train_loader = torch.utils.data.DataLoader(
          ColMNIST('data/mnist', train=True, download=True,
                                     transform=torchvision.transforms.Compose([#torchvision.transforms.Resize((224, 224)),
                                       torchvision.transforms.ToTensor(),
                                     ])),
          batch_size=batch_size_train, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
          ColMNIST('data/mnist', train=False, download=True,
                                     transform=torchvision.transforms.Compose([#torchvision.transforms.Resize((224, 224)),
                                       torchvision.transforms.ToTensor()
                                     ])),
          batch_size=batch_size_test, shuffle=True)

        examples = enumerate(self.train_loader)
        batch_idx, (example_data, example_targets) = next(examples)
        self.input_shape = example_data.shape[1:]
        self.model = AE(np.prod(self.input_shape)).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.n_conn_comp = n_conn_comp
        self.criterion = nn.MSELoss()


    def train(self, n_epochs):
        for e in range(1, n_epochs+1):
            loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.view(-1, np.prod(self.input_shape)).to(device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                lapl = laplacian(ws_to_adj([self.model.encoder_output_layer.weight, self.model.decoder_hidden_layer.weight]))
                self.ev, self.evec = torch.symeig((lapl), eigenvectors=True)
                block_reg = torch.sum(self.ev[:self.n_conn_comp])
                train_loss = self.criterion(outputs, data) + block_reg
                train_loss.backward(retain_graph=True)
                self.optimizer.step()
                loss += train_loss.item()
                if batch_idx%100 == 0:
                    print("Batch {}, loss {}".format(batch_idx+1, loss/(batch_idx+1)))
            loss = loss / len(self.train_loader)
            print("Train epoch {}, loss {}".format(e, loss))

import argparse
parser = argparse.ArgumentParser(description='Autoencoder on MNIST')
parser.add_argument('--epochs', type=int)
parser.add_argument('--connected_components', type=int)
args = parser.parse_args()

trainer = Trainer(args.connected_components)
trainer.train(args.epochs)
