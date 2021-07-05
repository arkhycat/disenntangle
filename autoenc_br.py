from col_mnist import ColMNIST
from split_dataset import SplitDS
from three_d_shapes_ds import ThreeDShapes
from disentangling_vae.utils.datasets import DSprites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from spectral_utils import *
from models import AE, block_regularizer, compute_layer_blocks_out

device = "cuda" if torch.cuda.is_available() else "cpu"

def clamp(x, minval=0, maxval=1):
    return max(min(x, maxval), minval)

class Trainer:
    def __init__(self, args, batch_size_train = 32, batch_size_test= 32, dataset='colmnist'):
        if args.dataset == 'colmnist':
            self.train_loader = torch.utils.data.DataLoader(
              ColMNIST('data/mnist', train=True, download=True,
                                         transform=torchvision.transforms.Compose([#torchvision.transforms.Resize((224, 224)),
                                           torchvision.transforms.ToTensor(),
                                         ])),
              batch_size=batch_size_train, shuffle=True, num_workers = args.n_workers)

            self.test_loader = torch.utils.data.DataLoader(
              ColMNIST('data/mnist', train=False, download=True,
                                         transform=torchvision.transforms.Compose([#torchvision.transforms.Resize((224, 224)),
                                           torchvision.transforms.ToTensor()
                                         ])),
              batch_size=batch_size_test, shuffle=True, num_workers = args.n_workers)
        elif args.dataset == 'split':
            self.train_loader = torch.utils.data.DataLoader(SplitDS(), batch_size=32, shuffle=True, num_workers = args.n_workers)
            self.test_loader = torch.utils.data.DataLoader(SplitDS(), batch_size=32, shuffle=True, num_workers = args.n_workers)
        elif args.dataset == 'dsprites':
            self.train_loader = torch.utils.data.DataLoader(
                                                  DSprites('data/sprites'),
                                                  batch_size=32, shuffle=True, num_workers = args.n_workers)
            self.test_loader = torch.utils.data.DataLoader(
                                                  DSprites('data/sprites'),
                                                  batch_size=32, shuffle=True, num_workers = args.n_workers)
        elif args.dataset == '3dshapes':
            self.train_loader = torch.utils.data.DataLoader(
                                                  ThreeDShapes(transform=torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), torchvision.transforms.Resize((32, 32)),
                                                    torchvision.transforms.ToTensor(),
                                                  ])),
                                                  batch_size=32, shuffle=True, num_workers = args.n_workers)
            self.test_loader = torch.utils.data.DataLoader(
                                                  ThreeDShapes(transform=torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), torchvision.transforms.Resize((32, 32)),
                                                    torchvision.transforms.ToTensor(),
                                                  ])),
                                                  batch_size=32, shuffle=True, num_workers = args.n_workers)


        examples = enumerate(self.train_loader)
        batch_idx, (example_data, example_targets) = next(examples)
        self.input_shape = example_data.shape[1:]
        self.model = AE(self.input_shape, bottleneck_size=args.bn_size, ncc=args.connected_components).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.n_conn_comp = args.connected_components
        self.bn_size = args.bn_size
        self.bn_size_reduction = 10
        self.criterion = nn.MSELoss()
        self.reg = args.regularizer
        self.batch_size = batch_size_train

    def relative_error(self, layer, blocks, examples):
        self.model.eval()
        relative_error = [None]*layer.out_features
        for c in range(self.n_conn_comp):
            mask = torch.zeros(layer.out_features, dtype=torch.float)
            mask[blocks==c] = 1
            layer.turn_output_neurons_off(mask)
            with torch.no_grad():
                reconstruction = trainer.model(examples)
            block_error = trainer.criterion(examples, reconstruction).item()
            layer.turn_all_output_neurons_on()

            for n in np.where(blocks!=c)[0]:
                mask = torch.zeros(layer.out_features, dtype=torch.float)
                mask[blocks==c] = 1
                mask[n] = 1
                #print(n)
                #plt.figure(figsize=(20, 20))
                #plt.imshow(mask.unsqueeze(0), cmap='magma')
                layer.turn_output_neurons_off(mask)

                with torch.no_grad():
                    reconstruction = trainer.model(examples)
                #plot_reconstruction(test_examples.detach(), reconstruction.detach().cpu())
                relative_error[n] = block_error - self.criterion(examples, reconstruction).item()
                layer.turn_all_output_neurons_on()
        return relative_error


    def neuron_wise_error(self, layer, blocks, examples):
        self.model.eval()
        relative_error = [None]*layer.out_features
        for n in range(layer.out_features):
            mask = torch.ones(layer.out_features, dtype=torch.float)
            mask[n] = 0
            layer.turn_output_neurons_off(mask)
            with torch.no_grad():
                reconstruction = trainer.model(examples)
            relative_error[n] = self.criterion(examples, reconstruction).item()
            layer.turn_all_output_neurons_on()
        return relative_error

    def neuron_wise_br(self, layer, blocks, examples):
        self.model.eval()
        relative_error = [None]*layer.out_features
        for n in range(layer.out_features):
            mask = torch.ones(layer.out_features, dtype=torch.bool, device=device)
            mask[n] = 0
            a_n = normalize_w(self.model.encoder_output_layer.weight[mask])
            _, s, _ = torch.svd(a_n)
            relative_error[n] = self.n_conn_comp - torch.sum(s[:self.n_conn_comp]).detach().cpu()
        return relative_error

    def train(self, n_epochs):
        blocks = None
        total_batches = len(self.train_loader)*n_epochs
        ee_coeff = 1
        ee_thr = 0
        mask = torch.ones(self.model.encoder_output_layer.weight.shape)
        do_rate = 0.5
        for e in range(1, n_epochs+1):
            loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.model.train()
                data = data.to(device)
                self.optimizer.zero_grad()
                global_batch_idx = (e-1)*len(self.train_loader)+batch_idx
                #ee_coeff = min(1, (global_batch_idx/total_batches)**2)
                ee_coeff = 1-clamp((global_batch_idx-ee_thr)/(total_batches-ee_thr))
                #ee_coeff = 1-clamp((global_batch_idx-ee_thr)/500)
                #outputs = self.model(data, blocks, 0.7*(1-clamp(ee_coeff, maxval=0.7)))
                #do_rate += 0.8/3600
                outputs = self.model(data, do_rate)
                rec_loss = self.criterion(outputs, data)
                if self.reg == "spectral":
                    lapl = laplacian(ws_to_adj([self.model.encoder_output_layer.weight, self.model.decoder_hidden_layer.weight]))
                    self.ev, self.evec = torch.symeig((lapl), eigenvectors=True)
                    block_reg = torch.sum(self.ev[:self.n_conn_comp])
                    train_loss = rec_loss + block_reg
                elif self.reg == "svd":
                    block_reg = block_regularizer(self.model.encoder_output_layer, self.n_conn_comp)
                    train_loss = rec_loss# + block_reg*(1-ee_coeff)*0.1
                    #blocks = compute_layer_blocks(self.model.encoder_output_layer, self.n_conn_comp)


                train_loss.backward(retain_graph=True)
                self.optimizer.step()
                loss += rec_loss.item()
                if batch_idx%100 == 0:
                    for batch_features in trainer.test_loader:
                        batch_features = batch_features[0]
                        test_examples = batch_features.to(device)
                        break
                    self.model.eval()
                    outputs = self.model(batch_features)
                    val_loss = self.criterion(outputs, batch_features)
                    print(("Batch {}, training loss {:.4f}, val loss {:.4f}, "
                            "block_reg {:.2f}, "
                            "dropout rate {:.2f}").format(batch_idx+1,
                            rec_loss.item(), val_loss,
                            block_reg.item(), do_rate))

                if (batch_idx)%(total_batches/(self.bn_size_reduction)) == 0:
                    #print(((len(self.train_loader)*n_epochs)/self.batch_size)/(self.final_bn_size))
                    re = self.neuron_wise_br(self.model.encoder_output_layer, blocks, test_examples)
                    removal_mask = torch.ones(self.model.encoder_output_layer.out_features, dtype=torch.bool)
                    #removal_mask[np.argmin(re)] = 0
                    print(re)
                    self.model.encoder_output_layer.remove_neurons_out(removal_mask)
                    self.model.decoder_hidden_layer.remove_neurons_in(removal_mask)
                    blocks = compute_layer_blocks_out(self.model.encoder_output_layer, self.n_conn_comp)
                    print("bn size:"+str(self.model.encoder_output_layer.out_features))

            loss = loss / len(self.train_loader)
            print("Train epoch {}, loss {}".format(e, loss))

import argparse
parser = argparse.ArgumentParser(description='Autoencoder on MNIST')
parser.add_argument('--epochs', type=int)
parser.add_argument('--connected_components', type=int)
parser.add_argument('--bn_size', type=int, default=100)
parser.add_argument('--n_workers', type=int, default=1)
parser.add_argument('--dataset')
parser.add_argument('--regularizer', default = "spectral")
args = parser.parse_args()

trainer = Trainer(args)
trainer.train(args.epochs)
