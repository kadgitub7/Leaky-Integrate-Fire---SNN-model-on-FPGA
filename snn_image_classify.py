import snntorch as snn
import torch
from torchvision import datasets, transforms
from snntorch import utils, spikegen
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML
import time
import random

def plot_mem(mem, title=False):
  if title:
    plt.title(title)
  plt.plot(mem)
  plt.xlabel("Time step")
  plt.ylabel("Membrane Potential")
  plt.xlim([0, 50])
  plt.ylim([0, 1])
  plt.show()

# training parameters
batch_size = 128 #number of sample per chunk
num_class = 10 # output classes (0-9)
dtype = torch.float # data type for the tensors
data_path='/tmp/data/mnist'

# Add transformation to dataset, resize to 28x28, grayscale, convert to tensors, and normalize RGB to 0-1
transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
'''
train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)

# subset of the dataset for initial testing
subset_size = 10 # dividing factor for the dataset size, e.g. 10 means 1/10th of the dataset will be used
train_dataset = utils.data_subset(train_dataset, subset_size)
print("training size =", len(train_dataset))
train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # DataLoader serves the whole data into chunks of batch_size, and shuffles the data for each epoch

# Rate Encoding Example
raw_vector = torch.ones(100)*0.5 # vector of 10 elements each with value 0.5
rate_encoded_vector = torch.bernoulli(raw_vector) # bernoulli takes the values in each element of the vector and that is the probability. It then uses that probability to generate 1 or 0
print('raw_vector =', raw_vector)
print('rate_encoded_vector =', rate_encoded_vector)
print(f'spiking rate: {rate_encoded_vector.sum()/len(rate_encoded_vector)*100:.2f} % of the time')

# Rate Encoding from snntorch
data = iter(train_dataset_loader)
data_it, target_it = next(data)

spike_gen = spikegen.rate(data_it, num_steps=100)
print(spike_gen.size())

# Leaky Integrate and Fire Neuron snntorch
print('Leaky Integrate and Fire')
time_step = 1e-3
R = 5
C = 1e-3

lif1 = snn.Lapicque(R=R, C=C, time_step=time_step)

mem = torch.ones(1) * 0.9  # U=0.9 at t=0
cur_in = torch.zeros(100, 1)  # I=0 for all t
spk_out = torch.zeros(1)  # initialize output spikes

mem_record = [mem]

for step in range(100):
  spk_out, mem = lif1(cur_in[step], mem)

  # Store recordings of membrane potential
  mem_record.append(mem)

# convert the list of tensors into one tensor
mem_record = torch.stack(mem_record)

# pre-defined plotting function
plot_mem(mem_record, "Lapicque's Neuron Model Without Stimulus")

'''

'''
### SNN single neuron
print('snn network')

lif = snn.Leaky(beta=0.9)
cur_in = torch.ones(1,1)*0.5
mem = torch.zeros(1,1)
mem_rec = []
spk_rec = []

for i in range(10):
    spk,mem = lif(cur_in, mem)
    mem_rec.append(mem)
    spk_rec.append(spk)
    random_int = random.randint(1,4)
    time.sleep(random_int)

print(mem_rec)
print(spk_rec)
'''

# network of SNN
beta = 0.8
input_layer = 784
hidden_layer = 1000
output_layer = 10

fc1 = torch.nn.Linear(input_layer, hidden_layer)
lif1 = snn.Leaky(beta=beta)
fc2 = torch.nn.Linear(hidden_layer, output_layer)
lif2 = snn.Leaky(beta=beta)

mem1 = lif1.init_leaky()
mem2 = lif2.init_leaky()

# record outputs
mem2_rec = []
spk1_rec = []
spk2_rec = []

spk_in = spikegen.rate_conv(torch.rand(200,784)).unsqueeze(1)
print(spk_in.size())

for step in range(200):
    cur1 = fc1(spk_in[step]) # post-synaptic current <-- spk_in x weight
    spk1, mem1 = lif1(cur1, mem1) # mem[t+1] <--post-syn current + decayed membrane
    cur2 = fc2(spk1)
    spk2, mem2 = lif2(cur2, mem2)

    mem2_rec.append(mem2)
    spk1_rec.append(spk1)
    spk2_rec.append(spk2)

# convert lists to tensors
mem2_rec = torch.stack(mem2_rec)
spk1_rec = torch.stack(spk1_rec)
spk2_rec = torch.stack(spk2_rec)



