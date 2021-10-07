import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import time
import random
import os
import argparse
import enum
import logging
from datetime import datetime
import json
from models import block_regularizer, compute_layer_blocks_in, compute_layer_blocks_out
from spectral_utils import normalize_w

from three_d_shapes_ds import ThreeDShapes
from col_mnist import ColMNIST
from models import DisentangledLinear, BlockDropout
import homebrew_vgg

dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d.%m.%Y(%H:%M:%S)")

class SupportedDatasets(enum.Enum):
    THREEDSHAPES = 0,
    COL_MNIST = 1,
    CIFAR10 = 2

parser = argparse.ArgumentParser()

# Common params
parser.add_argument("--dataset", type=str, choices=[ds.name for ds in SupportedDatasets], help="", 
                    default=SupportedDatasets.THREEDSHAPES.name)
parser.add_argument("--input_layer", type=str, help="Layer to disentangle", default='cl0')
parser.add_argument("--output_layer", type=str, help="Layer to disentangle", default='cl3')
parser.add_argument("--blocks", type=int, help="Number of blocks", default=4)
parser.add_argument("--layer_size", type=int, help="Size of the disentangled layer", default=400)
parser.add_argument("--prune_by", type=int, help="How many neurons we want to remove", default=200)
parser.add_argument("--data_dir", type=str, help="Directory to load data from", default='data')
parser.add_argument("--load_model", type=str, help="")
parser.add_argument("--save_dir", type=str, help="Directory to save models, logs and plots to", 
                    default=os.path.join("outputs", timestampStr))
parser.add_argument("--deterministic", dest="deterministic", action="store_true")
parser.add_argument("--no_dt_labels", dest="dt_labels", action="store_false")
parser.add_argument("--homebrew_model", dest="homebrew_model", action="store_true")
parser.add_argument("--filtered", help="Filter 3dshapes dataset (otherwise the decision tree labels are used)",
                     dest="filtered", action="store_true")
parser.add_argument("--gpus", type=str, help="", default=None)
parser.add_argument("--batch_size", type=int, help="", default=32)
parser.add_argument("--n_epochs", type=int, help="", default=30)
parser.add_argument("--img_size", type=int, help="", default=32)
parser.add_argument("--dropout_p", type=float, help="Probability of block dropout", default=0.5)
parser.add_argument("--optimizer", type=str, help="Optimizer", choices=["SGD", "Adam"], default="SGD")
parser.add_argument("--br_coef", type=float, help="Block regularizer coefficient", default=0)

args = parser.parse_args()  # important to put '' in Jupyter otherwise it will complain
parser.set_defaults(filtered=False, deterministic=False, dt_labels=True, homebrew_model=False)

config = dict()
# Wrapping configuration into a dictionary
for arg in vars(args):
    config[arg] = getattr(args, arg)
    
if not os.path.exists(config["save_dir"]):
    os.makedirs(config["save_dir"])
    
logging.basicConfig(filename=os.path.join(config["save_dir"], "run.log"), level=logging.DEBUG)

print("Saving and logging to {}".format(config['save_dir']))

if config['deterministic']:
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

device = "cpu"
if len(config["gpus"]) > 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=",".join(config["gpus"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #change to actual args
    
logging.info("Config {}".format(json.dumps(config, indent=4)))
logging.info("Using {}".format(device))

if config["dataset"] == SupportedDatasets.THREEDSHAPES.name:
    trainloader = torch.utils.data.DataLoader(
                                          ThreeDShapes(filename=os.path.join(config["data_dir"], "3dshapes.h5"),
                                                       transform=torchvision.transforms.Compose([
                                                           torchvision.transforms.ToPILImage(), 
                                                           torchvision.transforms.Resize((config["img_size"], config["img_size"])),
                                                           torchvision.transforms.ToTensor()]), 
                                                           filtered = config["filtered"],
                                                           dt_labels=config["dt_labels"]),
                                          batch_size=config["batch_size"], shuffle=True)

    testloader = torch.utils.data.DataLoader(
                                          ThreeDShapes(filename=os.path.join(config["data_dir"], "3dshapes.h5"),
                                                       transform=torchvision.transforms.Compose([
                                                           torchvision.transforms.ToPILImage(), 
                                                           torchvision.transforms.Resize((config["img_size"], config["img_size"])),
                                                           torchvision.transforms.ToTensor()]), 
                                                           filtered = config["filtered"],
                                                           dt_labels=config["dt_labels"]),
                                          batch_size=config["batch_size"], shuffle=True)

    if config["filtered"]:
        assert((not config["dt_labels"]))
        n_classes = 16
        def target_vec_to_class(vec):
            labels = (vec[:, 0] == 0).int()*(2**3) + (vec[:, 1] == 0).int()*(2**2) + (vec[:, 2] == 0)*2 + (vec[:, 4] == 0)
            return labels.long()
    else: #decision tree labels
        assert(config["dt_labels"])
        n_classes = 8
        def target_vec_to_class(tpl):
            latents, labels = tpl      
            return labels.long()
    
elif config["dataset"] == SupportedDatasets.COL_MNIST.name:
    trainloader = torch.utils.data.DataLoader(
      ColMNIST(os.path.join(config["data_dir"], "mnist"), train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(), torchvision.transforms.Resize((config["img_size"], config["img_size"]))
                                 ])),
      batch_size=config["batch_size"], shuffle=True)

    testloader = torch.utils.data.DataLoader(
      ColMNIST(os.path.join(config["data_dir"], "mnist"), train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(), torchvision.transforms.Resize((config["img_size"], config["img_size"]))
                                 ])),
      batch_size=config["batch_size"], shuffle=True)
    n_classes = 30
    def target_vec_to_class(tpl):
        (target, dclr_idx, bclr_idx) = tpl
        target += bclr_idx*10
        return target.long()

elif config["dataset"] == SupportedDatasets.CIFAR10.name:
    transform = transforms.Compose(
        [transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=config["data_dir"], train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"],
                                            shuffle=True)
    testset = torchvision.datasets.CIFAR10(root=config["data_dir"], train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config["batch_size"],
                                            shuffle=False)
    n_classes = 10
    def target_vec_to_class(x):
        return x

else:
    logging.error("Dataset not supported")
    
for data, target in testloader:
    break
img_shape = data.shape[1:]

if config["load_model"] is not None:
    vgg16 = torch.load(config["load_model"])
else:
    if config["homebrew_model"]:
        logging.error("Homebrew models are not pretrained")
        vgg16 = homebrew_vgg.vgg16(num_classes=n_classes, pretrained=False)
    else:
        vgg16 = models.vgg16(pretrained=True)
vgg16.to(device)

ncc = config["blocks"] #number of connected components

if config["input_layer"]=="cl3" and config["output_layer"]=="cl6":
    vgg16.classifier[3] = DisentangledLinear(vgg16.classifier[3].in_features, config["layer_size"]).to(device)
    vgg16.classifier[6] = DisentangledLinear(config["layer_size"], n_classes).to(device)
    vgg16.classifier[5] = BlockDropout(vgg16.classifier[6], ncc=ncc, p=config["dropout_p"], apply_to="in")
elif config["input_layer"]=="cl0" and config["output_layer"]=="cl3":
    # disentangle layers right after convolutions
    vgg16.classifier[0] = DisentangledLinear(vgg16.classifier[0].in_features, config["layer_size"]).to(device)
    vgg16.classifier[3] = DisentangledLinear(config["layer_size"], vgg16.classifier[3].out_features).to(device)
    vgg16.classifier[2] = BlockDropout(vgg16.classifier[3], ncc=ncc, p=config["dropout_p"], apply_to="in")
else:
    logging.error("Layer combination not supported")

for param in vgg16.parameters():
    param.requires_grad = True
    
logging.info(vgg16)

if config["optimizer"] == "Adam":
    optimizer = optim.Adam(vgg16.classifier.parameters(), lr=0.001)
else:
    optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001, momentum=0.9)
# loss function
criterion = nn.CrossEntropyLoss()

# validation function
def validate(model, test_dataloader):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    for int, data in enumerate(test_dataloader):
        data, target = data[0], data[1]
        target = target_vec_to_class(target)
        
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        
        val_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        val_running_correct += (preds == target).sum().item()
    
    val_loss = val_running_loss/len(test_dataloader.dataset)
    val_accuracy = 100. * val_running_correct/len(test_dataloader.dataset)
    logging.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')
    
    return val_loss, val_accuracy

def neuron_wise_br(model, layer, blocks, examples, ncc):
    model.eval()
    br_wo_neuron = [np.inf]*layer.out_features
    for n in range(layer.out_features):
        if layer.out_mask is not None:
            mask = torch.clone(layer.out_mask)
        else:
            mask = torch.ones(layer.out_features, dtype=torch.bool)
        if mask[n] > 0:
            mask[n] = 0
            a_n = normalize_w(layer.weight[mask])
            _, s, _ = torch.svd(a_n)
            br_wo_neuron[n] = ncc - torch.sum(s[:ncc]).detach().cpu()
    return br_wo_neuron

def plot_blocked_weights(layer):
    plt.figure(figsize=(20, 7))
    blocks_in = compute_layer_blocks_in(layer, ncc)
    blocks_out = compute_layer_blocks_out(layer, ncc)
    plt.imshow(layer.weight[np.argsort(blocks_out)][:, np.argsort(blocks_in)].cpu().detach().numpy())
    plt.show()
    
def prune(model, layer_out, layer_in, ncc):
    blocks = compute_layer_blocks_in(layer_out, ncc)
    for batch_features in testloader:
        batch_features = batch_features[0]
        test_examples = batch_features.to(device)
        break
    re = neuron_wise_br(model, layer_in, blocks, test_examples, ncc)
    removal_mask = layer_in.out_mask
    if removal_mask is None:
        removal_mask = torch.ones(layer_in.out_features, dtype=torch.bool)
    removal_mask[np.argmin(re)] = 0
    layer_out.turn_input_neurons_off(removal_mask)
    layer_in.turn_output_neurons_off(removal_mask)
    logging.info("Pruned to {} neurons".format(layer_out.in_mask.sum().item()))
    
def fit(model, train_dataloader, prune_every_n_steps):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in enumerate(train_dataloader):
        data, target = data[0], data[1]
        target = target_vec_to_class(target)
        
        #data = data.to(device)
        #target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if config["input_layer"]=="cl3":
            block_reg = block_regularizer(model.module.classifier[3], ncc)
        if config["input_layer"]=="cl0":
            block_reg = block_regularizer(model.module.classifier[0], ncc)
        loss = criterion(output.cpu(), target) + config["br_coef"]*block_reg
        #loss = block_reg
        train_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        train_running_correct += (preds.cpu() == target).sum().item()
        loss.backward()
        optimizer.step()
        if (i+1)%prune_every_n_steps == 0:
            logging.info("Block regularizer "+str(block_reg.item()))
            #plot_blocked_weights(vgg16.classifier[6])
            #plot_blocked_weights(vgg16.classifier[3])
            if config["input_layer"]=="cl3" and config["output_layer"]=="cl6" and \
                    (model.module.classifier[3].out_mask is None or model.module.classifier[3].out_mask.sum()>model.module.classifier[3].out_features-config["prune_by"]):
                prune(model.module, model.module.classifier[6], model.module.classifier[3], ncc)
            if config["input_layer"]=="cl0" and config["output_layer"]=="cl3" and \
                    (model.module.classifier[0].out_mask is None or model.module.classifier[0].out_mask.sum()>model.module.classifier[0].out_features-config["prune_by"]):
                prune(model.module, model.module.classifier[3], model.module.classifier[0], ncc)
            
    train_loss = train_running_loss/len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(train_dataloader.dataset)
    logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')
    
    return train_loss, train_accuracy, block_reg.item()

n_epochs = config["n_epochs"]
total_batches = len(trainloader)*n_epochs
layer_size_reduction = config["prune_by"]
prune_every_n_steps = int(round(total_batches/(layer_size_reduction)))

vgg16_parallel = nn.DataParallel(vgg16, device_ids = [0])

train_loss , train_accuracy = [], []
val_loss , val_accuracy, br = [], [], []

start = time.time()
for epoch in range(n_epochs):
    start_e = time.time()
    logging.info("Epoch {}".format(epoch))
    train_epoch_loss, train_epoch_accuracy, block_reg = fit(vgg16_parallel, trainloader, prune_every_n_steps)
    val_epoch_loss, val_epoch_accuracy = validate(vgg16_parallel, testloader)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    br.append(block_reg)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    torch.save(vgg16_parallel.module, os.path.join(config["save_dir"], "model.pt"))
    end_e = time.time()
    logging.info('Epoch {} took {} minutes '.format(epoch+1, (end_e-start_e)/60))
    
end = time.time()
logging.info('{} minutes in total'.format((end-start)/60))

plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='green', label='train accuracy')
plt.plot(val_accuracy, color='blue', label='validataion accuracy')
plt.legend()
plt.savefig(os.path.join(config["save_dir"], 'accuracy.png'))
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(br, color='red', label='block regularizer')
plt.legend()
plt.savefig(os.path.join(config["save_dir"], 'br.png'))
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.legend()
plt.savefig(os.path.join(config["save_dir"], 'loss.png'))
plt.show()

with open(os.path.join(config["save_dir"], 'config.json'), 'w') as f:
    json.dump(config, f)