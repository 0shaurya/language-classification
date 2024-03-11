import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time

######################
# Constants
######################

DATASET_SIZE_PER_LANG = 15000
BATCH_SIZE = 256
LEARNING_RATE = 0.003
SPLIT = (80, 15, 5) # train, validation, test
NUM_EPOCHS = 100
SEED = 0
WEIGHT_DECAY = 0

torch.manual_seed(SEED)
start_time = time.time()
print('Loading data...')

######################
# Data
######################

def load_csvs(csvs):
    datasets = []
    for i in range(len(csvs)):
        tensor = torch.tensor(pd.read_csv(csvs[i]).values)
        tensor = torch.cat((tensor, torch.full((tensor.shape[0], 1), float(i))), dim=1)
        tensor = tensor[torch.randperm(tensor.shape[0])]
        tensor = tensor[:DATASET_SIZE_PER_LANG]
        datasets.append(tensor)

    tensor = torch.cat(tuple(datasets), dim = 0)
    tensor = tensor[torch.randperm(tensor.shape[0])]

    return tensor

Xy = load_csvs(['english.csv',
                # 'spanish.csv'
                # 'japanese.csv'])
                'russian.csv'])
X, y = torch.split(Xy, [432,1], dim=1)
y = y.squeeze().long()

######################
# Dataset Initialization
######################

class LangDataset(Dataset):
    def __init__(self, dataset_type):
        split1 = int(X.shape[0]*SPLIT[0]/100)
        split2 = int(X.shape[0]*(SPLIT[0] + SPLIT[1])/100)
        if dataset_type == 'train':
            self.X = X[:split1]
            self.y = y[:split1]
        elif dataset_type == 'valid':
            self.X = X[split1:split2]
            self.y = y[split1:split2]
        elif dataset_type == 'test':
            self.X = X[split2:]
            self.y = y[split2:]

    def __getitem__(self, index):
        X_sample = self.X[index]
        y_sample = self.y[index]
        return X_sample, y_sample

    def __len__(self):
        return self.y.shape[0]


train = LangDataset('train')
valid = LangDataset('valid')
test = LangDataset('test')

train_loader = DataLoader(dataset=train,
                        batch_size=BATCH_SIZE,
                        drop_last=True,
                        shuffle=True,
                        num_workers=0)
valid_loader = DataLoader(dataset=valid,
                        batch_size=BATCH_SIZE,
                        drop_last=False,
                        shuffle=True,
                        num_workers=0)
test_loader = DataLoader(dataset=test,
                        batch_size=BATCH_SIZE,
                        drop_last=False,
                        shuffle=True,
                        num_workers=0)
print(f'Data Loaded ({(time.time()-start_time):.1f}s)')

def find_accuracy(data_loader, model):
    correct = 0
    total = 0
    
    for features, targets in data_loader:
        logits = model(features)
        _, predicted = torch.max(logits, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    accuracy = correct / total
    return accuracy


class network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(432, 15), torch.nn.ReLU(), torch.nn.BatchNorm1d(15), torch.nn.Dropout(.5),
            torch.nn.Linear(15, 15),  torch.nn.ReLU(), torch.nn.BatchNorm1d(15),
            torch.nn.Linear(15, 2))

    def forward(self, x):
        return self.model(x)

######################
# Running Model
######################

model = network()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

start_time = time.time()
epochs = NUM_EPOCHS
losses = []
train_accs = []
valid_accs = []
for epoch in range(epochs):

    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        logits = model(features)
        loss = F.cross_entropy(logits, targets)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        print(f"Epoch [{epoch+1}/{epochs}] | Batch {batch_idx+1} | Loss: {loss.item():.3f}")
    
    model.eval()
    with torch.no_grad():
        train_acc = find_accuracy(train_loader, model)
        valid_acc = find_accuracy(valid_loader, model)
        
        train_accs.append(train_acc * 100)
        valid_accs.append(valid_acc * 100)

    print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_accs[-1]:.2f} | Valid Acc: {valid_accs[-1]:.2f} | Loss: {losses[-1]:.4f} | Time Elapsed: {(time.time() - start_time):.1f}s")


model.eval()
with torch.no_grad():
    test_acc = find_accuracy(test_loader, model)

######################
# Plotting
######################

print(f"Test Acc: {test_acc:.4f}")


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Training')
plt.plot(valid_accs, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

LOSS_CURVE_SMOOTHING = 10
smoothed = []
for i in range(1, len(losses)):
    smoothed.append(sum(losses[max(i-LOSS_CURVE_SMOOTHING, 0):i]) / len(losses[max(i-LOSS_CURVE_SMOOTHING, 0):i]))

plt.subplot(1, 2, 2)
plt.plot(losses, label='Loss', c='green')
plt.plot(smoothed, label='Smoothed', c='purple')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.show()