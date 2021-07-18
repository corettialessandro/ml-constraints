import numpy as np
import matplotlib.pyplot as plt

import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class AE(nn.Module):
    def __init__(self):
        super(AE,  self).__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=784, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=16, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=784),
            nn.ReLU()
        )

    def forward(self, x):
        orig_shape = x.shape
        x_flat = self.flatten(x)
        x_encoded = self.encoder(x_flat)
        x_decoded = self.decoder(x_encoded)
        return x_decoded.reshape(orig_shape)

def train(dataloader, neural_network, loss_fn, optimizer):

    size = len(dataloader.dataset)

    for batch, (sample, label) in enumerate(dataloader):
        reconstructed = neural_network(sample)
        loss = loss_fn(reconstructed, sample)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(sample)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return

def test(dataloader, neural_network, loss_fn):

    num_batches = len(dataloader)

    test_loss = 0
    with torch.no_grad():
        for sample, label in dataloader:
            reconstructed = neural_network(sample)
            test_loss += loss_fn(reconstructed, sample).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    return

def reconstruct(sample, neural_network):

    with torch.no_grad():
        return neural_network(sample)

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    AE = AE().to(device)
    # print(AE)

    learning_rate = 1.e-3

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(AE.parameters(), lr = learning_rate)

    epochs = 5
    for epoch in range(epochs):

        print(f"Epoch {epoch+1}\n-------------------------------")

        train(train_dataloader, AE, loss_fn, optimizer)
        test(test_dataloader, AE, loss_fn)

    print("Training Complete!")

    sample_norm = torch.randn([1,28,28])
    generated_norm = reconstruct(sample_norm, AE)
    reconstructed_norm = reconstruct(generated_norm, AE)

    sample_unif = torch.rand([1,28,28])
    generated_unif = reconstruct(sample_unif, AE)
    reconstructed_unif = reconstruct(generated_unif, AE)

    orig_img_norm = sample_norm[0].squeeze()
    gen_img_norm = generated_norm[0].squeeze()
    rec_img_norm = reconstructed_norm[0].squeeze()

    orig_img_unif = sample_unif[0].squeeze()
    gen_img_unif = generated_unif[0].squeeze()
    rec_img_unif = reconstructed_unif[0].squeeze()

    f, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, squeeze=True)
    ax[0,0].imshow(orig_img_norm, cmap="gray")
    ax[0,1].imshow(gen_img_norm, cmap="gray")
    ax[0,2].imshow(rec_img_norm, cmap="gray")
    ax[1,0].imshow(orig_img_unif, cmap="gray")
    ax[1,1].imshow(gen_img_unif, cmap="gray")
    ax[1,2].imshow(rec_img_unif, cmap="gray")
    plt.show()
