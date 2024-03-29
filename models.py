import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from spectral_utils import normalize_w, blocks_from_svd
from math import floor

#device = "cuda" if torch.cuda.is_available() else "cpu"

def block_dropout_mask(blocks, prob):
    #low prob -> low probability of turning off
    mask = torch.zeros(len(blocks), dtype=torch.bool)
    first_mask = torch.rand(((blocks==0).sum(), )) > prob
    mask[blocks==0] = first_mask
    p = first_mask.sum()/len(first_mask)
    for i in range(1, max(blocks)+1):
        new_mask = torch.zeros(((blocks==i).sum(), ), dtype=torch.bool)
        new_mask[:int(p*len(new_mask))] = 1
        mask[blocks==i] = new_mask[torch.randperm(len(new_mask))]

    return mask.float()

def block_dropout(input, blocks, p):
    return input*block_dropout_mask(blocks, p).to(input.get_device())/(1-p)

def get_masked_weight(layer):
    masked_weight = layer.weight
    if layer.in_mask is not None:
        masked_weight = layer.weight[:, layer.in_mask]
    if layer.out_mask is not None:
        masked_weight = masked_weight[layer.out_mask]
    return masked_weight

def layer_svd(layer):
    masked_weight = get_masked_weight(layer)
    a_n = normalize_w(masked_weight) #TODO: not sure if this works directly
    return torch.svd(a_n)

def compute_layer_blocks_out(layer, ncc):
    #blocks are computer on the output because the weight matrix is transposed
    u, _, _ = layer_svd(layer)
    blocks = -torch.ones(layer.out_features, dtype=torch.int) #pruned neurons will have value of -1
    w = get_masked_weight(layer)
    d1_inv_root = torch.diag(1/torch.sqrt(torch.abs(w).sum(1)))
    masked_blocks = blocks_from_svd(torch.matmul(d1_inv_root, u), ncc)
    blocks[layer.out_mask] = torch.tensor(masked_blocks, dtype=torch.int)  #return blocking results back to unmasked shape
    return blocks

def compute_layer_blocks_in(layer, ncc):
    #blocks are computer on the input because the weight matrix is transposed
    _, _, v = layer_svd(layer)
    blocks = -torch.ones(layer.in_features, dtype=torch.int) #pruned neurons will have value of -1
    w = get_masked_weight(layer)
    d2_inv_root = torch.diag(1/torch.sqrt(torch.abs(w).sum(0)))
    masked_blocks = blocks_from_svd(torch.matmul(d2_inv_root, v), ncc)
    blocks[layer.in_mask] = torch.tensor(masked_blocks, dtype=torch.int) #return blocking results back to unmasked shape
    return blocks

def block_regularizer(layer, ncc):
    _, s, _ = layer_svd(layer)
    return ncc - torch.sum(s[:ncc])

class BlockDropout(nn.Dropout):
    def __init__(self, layer, ncc, p=0.5, apply_to="out"):
        super().__init__()
        self.layer = layer
        self.n_conn_comp = ncc
        self.apply_to = apply_to
        self.p = p

    def forward(self, input, do_rate=0):
        if do_rate == 0:
            do_rate = self.p
        if self.training:
            if self.n_conn_comp > 1:
                if self.apply_to=="out":
                    blocks = compute_layer_blocks_out(self.layer, self.n_conn_comp)
                else:
                    blocks = compute_layer_blocks_in(self.layer, self.n_conn_comp)
                #print(blocks.shape)
                #blocks = np.zeros(input.shape[1])
                #blocks[:int(input.shape[0]/2)] = 1

                #print(blocks.shape)
                return block_dropout(input, blocks, do_rate)
            return F.dropout(input, do_rate, self.training)
        else:
            return input

class DisentangledLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.turn_all_input_neurons_on()
        self.turn_all_output_neurons_on()

    def remove_neurons_in(self, mask):
        """ deletes neurons from input layer,
        0s in the mask indicate the neurons being removed """

        assert(len(mask) == self.weight.shape[1])
        self.weight = nn.Parameter(self.weight.clone().detach()[:, mask])
        self.in_features = mask.sum().item()
        self.in_mask = self.in_mask[mask]

    def remove_neurons_out(self, mask):
        """ deletes neurons from input layer,
        0s in the mask indicate the neurons being removed """

        assert(len(mask) == self.weight.shape[0])
        self.weight = nn.Parameter(self.weight.clone().detach()[mask])
        self.bias = nn.Parameter(self.bias.clone().detach()[mask])
        self.out_features = mask.sum().item()
        self.out_mask = self.out_mask[mask]

    def collapse_neurons_in(self, mask):
        """ collapses several neurons into one,
        same values in the mask indicate the neurons
        the averaging is performed on
        neurons with 0 in the mask are left untouched"""
        assert(len(mask) == self.weight.shape[1])
        new_weight = self.weight
        first_n_idxs = np.array([np.where(mask==i)[0][0] for i in range(1, max(mask)+1)])
        first_n_idxs = np.concatenate((first_n_idxs, np.where(mask==0)[0]), axis=0)
        first_n_idxs.sort()
        for i in range(1, max(mask)+1):
            n_idx = np.where(mask==i)[0][0]
            new_weight[:, n_idx] = self.weight[:, mask==i].mean(dim=1)
        self.weight = nn.Parameter(new_weight[:, first_n_idxs])
        self.in_features = len(first_n_idxs)

    def collapse_neurons_out(self, mask):
        """ collapses several neurons into one,
        same values in the mask indicate the neurons
        the averaging is performed on
        neurons with 0 in the mask are left untouched"""

        assert(len(mask) == self.weight.shape[0])
        new_weight = self.weight
        first_n_idxs = np.array([np.where(mask==i)[0][0] for i in range(1, max(mask)+1)])
        first_n_idxs = np.concatenate((first_n_idxs, np.where(mask==0)[0]), axis=0)
        first_n_idxs.sort()
        for i in range(1, max(mask)+1):
            n_idx = np.where(mask==i)[0][0]
            new_weight[n_idx] = self.weight[mask==i].mean(dim=0)
        self.weight = nn.Parameter(new_weight[first_n_idxs])
        self.out_features = len(first_n_idxs)

    def turn_input_neurons_off(self, mask):
        """ 0s in the mask indicate the neurons being turned off """
        #self.in_mask = torch.tensor(mask, requires_grad=False).to(self.weight.get_device())
        self.in_mask = mask.clone().detach().requires_grad_(False).to(self.weight.get_device())

    def turn_output_neurons_off(self, mask):
        """ 0s in the mask indicate the neurons being turned off """
        #self.out_mask = torch.tensor(mask, requires_grad=False).to(self.weight.get_device())
        self.out_mask = mask.clone().detach().requires_grad_(False).to(self.weight.get_device())

    def turn_all_input_neurons_on(self):
        self.in_mask = None

    def turn_all_output_neurons_on(self):
        self.out_mask = None

    def forward(self, input):
        if self.in_mask is not None:
            input = self.in_mask.to(input.get_device())*input
        if self.out_mask is not None:
            return self.out_mask.to(input.get_device()) * super().forward(input)
        else:
            return super().forward(input)


class AE(nn.Module):
    def __init__(self, input_shape, bottleneck_size=10, hidden_size=1024, ncc=1):
        super().__init__()

        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.n_conn_comp = ncc
        self.encoder_0 = nn.Conv2d(input_shape[0], 6, kernel_size=5)
        self.encoder_1 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))

        self.enc_shape = (16, floor((self.input_shape[-2]-4*2)), floor((self.input_shape[-1]-4*2)))
        self.enc_size = int(np.prod(self.enc_shape))
        #print(self.enc_size, self.enc_shape)
        self.encoder_hidden_layer = nn.Linear(in_features=self.enc_size, out_features=self.hidden_size)
        self.encoder_output_layer = DisentangledLinear(in_features=self.hidden_size, out_features=bottleneck_size)
        self.block_dropout_1 = BlockDropout(self.encoder_output_layer, ncc=self.n_conn_comp)
        self.decoder_hidden_layer = DisentangledLinear(in_features=bottleneck_size, out_features=self.hidden_size)
        self.block_dropout_2 = BlockDropout(self.decoder_hidden_layer, ncc=self.n_conn_comp)
        self.decoder_output_layer = nn.Linear(in_features=self.hidden_size, out_features=self.enc_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,input_shape[0],kernel_size=5),
            nn.ReLU(True))

        self.mask = None
        self.original_w = None

    def forward(self, x, blocks=None, do_rate=0.2, ee_coeff=1.0):
        if self.mask is not None:
            device = self.encoder_output_layer.weight.get_device()
            self.encoder_output_layer.weight = nn.Parameter(self.encoder_output_layer.weight.to(device) * self.mask.to(device))
            self.decoder_hidden_layer.weight = nn.Parameter(self.decoder_hidden_layer.weight.to(device) * torch.transpose(self.mask, 1, 0).to(device))

        self.layer_0_out = self.encoder_0(x)
        x = self.encoder_1(self.layer_0_out)
        #print(x.shape)
        x = x.view(-1, self.enc_size)
        #print(x.shape)
        x = F.dropout(torch.tanh(self.encoder_hidden_layer(x)), p=0.2)
        #x = torch.tanh(self.encoder_hidden_layer(x))
        self.embedding = torch.tanh(self.encoder_output_layer(x))
        #x = F.dropout(self.embedding)
        x = self.embedding
        #if blocks is not None and self.training:
        x = self.block_dropout_1(x, do_rate)

        #print(x.shape)
        #x = self.block_dropout_2(torch.tanh(self.decoder_hidden_layer(x)), do_rate)
        x = torch.tanh(self.decoder_hidden_layer(x))
        x = F.dropout(torch.tanh(self.decoder_output_layer(x)), p=0.2)
        #print(x.shape)
        x = x.view(-1, *self.enc_shape)
        #print(x.shape)
        x = self.decoder(x)
        #print(x.shape)

        return x


class AE_baseline(nn.Module):
    def __init__(self, input_shape, bottleneck_size=10, hidden_size=1024, ncc=1):
        super().__init__()

        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.encoder_0 = nn.Conv2d(input_shape[0], 6, kernel_size=5)
        self.encoder_1 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))

        self.enc_shape = (16, floor((self.input_shape[-2]-4*2)), floor((self.input_shape[-1]-4*2)))
        self.enc_size = int(np.prod(self.enc_shape))
        #print(self.enc_size, self.enc_shape)
        self.encoder_hidden_layer = nn.Linear(in_features=self.enc_size, out_features=self.hidden_size)
        self.encoder_output_layer = DisentangledLinear(in_features=self.hidden_size, out_features=bottleneck_size)
        self.decoder_hidden_layer = DisentangledLinear(in_features=bottleneck_size, out_features=self.hidden_size)
        #self.encoder_output_layer = nn.Linear(in_features=self.hidden_size, out_features=bottleneck_size)
        #self.decoder_hidden_layer = nn.Linear(in_features=bottleneck_size, out_features=self.hidden_size)
        self.decoder_output_layer = nn.Linear(in_features=self.hidden_size, out_features=self.enc_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,input_shape[0],kernel_size=5),
            nn.ReLU(True))


    def forward(self, x, blocks=None, do_rate=0.2, ee_coeff=1.0):
        self.layer_0_out = self.encoder_0(x)
        x = self.encoder_1(self.layer_0_out)
        #print(x.shape)
        x = x.view(-1, self.enc_size)
        #print(x.shape)
        x = F.dropout(torch.tanh(self.encoder_hidden_layer(x)), p=0.2)
        #x = torch.tanh(self.encoder_hidden_layer(x))
        self.embedding = torch.tanh(self.encoder_output_layer(x))
        #x = F.dropout(self.embedding)
        x = self.embedding
        #if blocks is not None and self.training:
        x = F.dropout(x, p=0.5)

        #print(x.shape)
        #x = self.block_dropout_2(torch.tanh(self.decoder_hidden_layer(x)), do_rate)
        x = F.dropout(torch.tanh(self.decoder_hidden_layer(x)), p=0.5)
        x = F.dropout(torch.tanh(self.decoder_output_layer(x)), p=do_rate)
        #print(x.shape)
        x = x.view(-1, *self.enc_shape)
        #print(x.shape)
        x = self.decoder(x)
        #print(x.shape)

        return x
