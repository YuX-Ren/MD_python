import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.utils.data as data
print("Using torch", torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import sys 
import os

import MDAnalysis as mda

from GIN.GIN import calculate_forces

pdb_file = "ala15.pdb"
psf_file = "ala15.psf"

u = mda.Universe(psf_file, pdb_file)
layers = 3
dim = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Force_position(data.Dataset):
    def __init__(self, path, transform=None):
        pass
    def __len__(self):
        pass
    def __getitem__(self, index):
        pass

def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

model = calculate_forces(u, layers, dim).to(device)
# model.load_state_dict(torch.load("GIN_model.pt"))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()
dataset = Force_position(psf_file, transform=None)
size = len(dataset)
train_dataset, test_dataset = data.random_split(dataset,[int(size*0.9),size-int(size*0.9)])
train_loader = data.DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = data.DataLoader(test_dataset,batch_size=32,shuffle=True)

train(model, train_loader, optimizer, loss_function, 10)

torch.save(model.state_dict(), "GIN_model.pt")
