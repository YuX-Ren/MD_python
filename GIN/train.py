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
torch.manual_seed(42)
import MDAnalysis as mda

from GIN import calculate_forces


pdb_file = "ala15.pdb"
psf_file = "ala15.psf"

u = mda.Universe(psf_file, pdb_file)
layers = 3
dim = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Force_position(data.Dataset):
    def __init__(self, path, transform=None):
        self.data = torch.load(path+'/positions.pt').float()
        self.target = torch.load(path+'/forces.pt').float()
    def __len__(self):
        return self.data.size(0)
    def __getitem__(self, index):
        return self.data[index], self.target[index]

def train(model, train_loader,test_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(1, num_epochs+1):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for batch_idx, (data, target) in progress_bar:
            bsz = data.size(0)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            training_loss = loss.item()/torch.linalg.norm(target, dim=1).mean()
            progress_bar.set_postfix({'training_loss': '{:.6f}'.format(training_loss)})
        test(model, test_loader, criterion)
with torch.no_grad():
    def test(model, test_loader, criterion):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # print(output)
                test_loss += criterion(output, target).item()/ torch.linalg.norm(target, dim=1).mean()
            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f})\n'.format(
                test_loss*1000))

model = calculate_forces(u, layers, dim).to(device)
# model.load_state_dict(torch.load("GIN_model.pt"))
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001,alpha=0.99)
loss_function = nn.MSELoss()
dataset = Force_position('./GIN', transform=None)
size = len(dataset)
train_dataset, test_dataset = data.random_split(dataset,[int(size*0.9),size-int(size*0.9)])
train_loader = data.DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = data.DataLoader(test_dataset,batch_size=32,shuffle=True)

train(model, train_loader, test_loader ,optimizer, loss_function, 10)
test(model, test_loader, loss_function)
# torch.save(model.state_dict(), "./GIN_model_SGD.pt")
