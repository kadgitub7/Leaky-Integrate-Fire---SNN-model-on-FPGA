import snntorch as snn
import torch
from snntorch import spikegen
from torch.utils.data import DataLoader
import numpy as np
import wfdb

# Training parameters
batch_size = 128 
dtype = torch.float 

# Dataset: Download database
# wfdb.dl_database('mitdb', dl_dir='./mitdb_data')

# Custom PyTorch Dataset for Multi-Lead MIT-BIH
class MITBIHDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        num_beats = data.shape[0]
        flattened_data = data.reshape(num_beats, -1) # Flatten 198 x num_leads
        
        self.data = torch.tensor(flattened_data, dtype=torch.float32)

        # Normalize individual beats to [0, 1] range for latency encoding
        mins = self.data.min(dim=1, keepdim=True).values
        maxs = self.data.max(dim=1, keepdim=True).values
        self.data = (self.data - mins) / (maxs - mins + 1e-8)

        self.targets = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

import os

# Define standard patient-wise splits (DS1 for training, DS2 for testing)
train_record_names = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119', 
    '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230'
]

test_record_names = [
    '100', '103', '105', '111', '113', '117', '121', '202', '210', '212', 
    '213', '214', '219', '221', '222', '228', '231', '232', '233', '234'
]

# AAMI labels
N = ['N', 'L', 'R', 'e', 'j']
S = ['A', 'a', 'J', 'S']
V = ['V', 'E']
F = ['F']
Q = ['/', 'f', 'Q']

aami_map = {}
for sym in N: aami_map[sym] = 0
for sym in S: aami_map[sym] = 1
for sym in V: aami_map[sym] = 2
for sym in F: aami_map[sym] = 3
for sym in Q: aami_map[sym] = 4

# Download the entire database (this downloads once and skips if already present)
# print("Downloading/verifying MIT-BIH database files...")
# wfdb.dl_database('mitdb', dl_dir='./mitdb_data')

win_left = 90    # ~250ms before R-peak
win_right = 108  # ~300ms after R-peak

beats = []
beat_labels = []

def extract_beats_from_records(rec_list):
    beats, labels = [], []
    for rec_id in rec_list:
        record_path = os.path.join('./mitdb_data', rec_id)
        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            signals = record.p_signal
            
            for idx, label in zip(annotation.sample, annotation.symbol):
                if idx - win_left >= 0 and idx + win_right < len(signals) and label in aami_map:
                    beat_segment = signals[idx - win_left : idx + win_right, :]
                    beats.append(beat_segment)
                    labels.append(label)
        except Exception as e:
            print(f"Skipping {rec_id}: {e}")
    return np.array(beats), labels

print("Extracting Training Patients...")
train_beats, train_raw_labels = extract_beats_from_records(train_record_names)

print("Extracting Testing Patients...")
test_beats, test_raw_labels = extract_beats_from_records(test_record_names)

# Combine labels to map unique classes properly across both sets
num_class = 5
train_numeric_labels = [aami_map[l] for l in train_raw_labels]
test_numeric_labels = [aami_map[l] for l in test_raw_labels]

# Create separate Dataset instances
train_dataset = MITBIHDataset(train_beats, train_numeric_labels)
test_dataset = MITBIHDataset(test_beats, test_numeric_labels)

# Directly create your DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Determine number of leads and classes from our actual patient arrays
num_leads = train_beats.shape[2]
print(f"Detected {num_leads} lead(s) across dataset.")
print(f"Detected classes: 5")
print(f"Total classes: {num_class}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

class Net(torch.nn.Module):
    def __init__(self):
       super().__init__()
       self.fc1 = torch.nn.Linear(input_layer, hidden_layer)
       self.lif1 = snn.Leaky(beta=beta)
       self.fc2 = torch.nn.Linear(hidden_layer, output_layer)
       self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
       mem1 = self.lif1.init_leaky()
       mem2 = self.lif2.init_leaky()

       mem2_rec = []
       spk2_rec = []

       for step in range(num_steps):
          cur1 = self.fc1(x[step]) 
          spk1, mem1 = self.lif1(cur1, mem1) 
          cur2 = self.fc2(spk1)
          spk2, mem2 = self.lif2(cur2, mem2)

          mem2_rec.append(mem2)
          spk2_rec.append(spk2)

       return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

def print_batch_accuracy(data, targets, train=False):
    spike_data = spikegen.rate(data, num_steps = num_steps, gain = 0.5)
    # spike_data = spikegen.latency(data, num_steps=num_steps, tau=5.0, threshold=0.01, normalize=True)

    
    output, _ = net(spike_data)
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

# Network hyperparameters updated for multi-lead input size
beta = 0.8
input_layer = 198 * num_leads  
hidden_layer = 256      
output_layer = num_class 

num_steps = 25          

net = Net().to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9,0.999))          

num_epochs = 30
loss_hist = []
test_loss_hist = []
counter = 0

# Initialize test iterator safely *before* the training loop starts
test_iter = iter(test_loader)

# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0

    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)

        spike_data = spikegen.rate(data, num_steps = num_steps, gain = 0.5)
        # spike_data = spikegen.latency(data, num_steps=num_steps, tau=5.0, threshold=0.01, normalize=True)

        net.train()
        spk_rec, mem_rec = net(spike_data)

        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        optimizer.zero_grad()
        # surrogate gradient in backpropogation takes the step wise activation function and approximates
        loss_val.backward()
        optimizer.step()

        loss_hist.append(loss_val.item())

        # Test set evaluation per iteration
        with torch.no_grad():
            net.eval()
            try:
                test_data, test_targets = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                test_data, test_targets = next(test_iter)

            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            test_spike_data = spikegen.rate(test_data, num_steps = num_steps, gain = 0.5)
            # test_spike_data = spikegen.latency(data, num_steps=num_steps, tau=5.0, threshold=0.01, normalize=True)
        
            test_spk, test_mem = net(test_spike_data)

            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            if counter % 50 == 0:
                train_printer()
            counter += 1
            iter_counter += 1

# Evaluation Phase
total = 0
correct = 0
all_preds = []
all_targets = []

with torch.no_grad():
  net.eval()
  for data, targets in test_loader:
    data = data.to(device)
    targets = targets.to(device)

    test_spike_data = spikegen.rate(data, num_steps = num_steps, gain = 0.5)
    # test_spike_data = spikegen.latency(data, num_steps=num_steps, tau=5.0, threshold=0.01, normalize=True)

    test_spk, _ = net(test_spike_data)

    _, predicted = test_spk.sum(dim=0).max(1)
    total += targets.size(0)
    correct += (predicted == targets).sum().item()
    all_preds.extend(predicted.cpu().numpy())
    all_targets.extend(targets.cpu().numpy())

print(f"Number correct = {correct}/{total}")
print(f"Percentage correct = {(correct/total * 100):.2f}%")

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(all_targets, all_preds, target_names=['N','S','V','F','Q']))
print(confusion_matrix(all_targets, all_preds))