import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from col_mnist import ColMNIST

plt.rcParams["axes.grid"] = False
device = "cuda" if torch.cuda.is_available() else "cpu"

def show_dataset_examples(trainer):
    def imshow(img):
        #img = img / 2 + 0.5     # unnormalize
        npimg = img.cpu().numpy()
        plt.rcParams['xtick.bottom'] = False
        plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['ytick.left'] = False
        plt.rcParams['ytick.labelleft'] = False
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    dataiter = iter(trainer.test_loader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    img_shape = images[0].shape
    print("Image shape: {}".format(img_shape))

def plot_reconstruction(test_examples, reconstruction):
    with torch.no_grad():
        number = 10
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(np.transpose(test_examples[index].cpu().numpy(), (1, 2, 0)))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, number, index + 1 + number)
            plt.imshow(np.clip(np.transpose(reconstruction[index].numpy(), (1, 2, 0)), 0, 1))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

def show_reconstruction(test_examples, trainer):
    with torch.no_grad():
        trainer.model.eval()
        reconstruction = trainer.model(test_examples.cuda()).cpu()
    plot_reconstruction(test_examples, reconstruction)

def get_test_sample(trainer):
    for batch_features in trainer.test_loader:
        batch_features = batch_features[0]
        test_examples = batch_features
        return test_examples
