import snntorch as snn
import torch
from torchvision import datasets, transforms
from snntorch import utils, spikegen
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML
import numpy as np

# training parameters
batch_size = 128 #number of sample per chunk
num_class = 10 # output classes (0-9)
dtype = torch.float # data type for the tensors
data_path='/tmp/data/mnist'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Add transformation to dataset, resize to 28x28, grayscale, convert to tensors, and normalize RGB to 0-1
transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

class LeakySurrogate(torch.nn.Module):
    def __init__(self, beta, threshold = 1.0):
      super(LeakySurrogate, self).__init__()
      # Initializing parameters, mostly the same
      self.beta = beta
      self.threshold = threshold
      self.spike_gradient = self.ATan.apply

    def forward(self, input_, mem):
      spk = self.spike_gradient((mem-self.threshold))
      reset = (self.beta * spk * self.threshold).detach()
      mem = self.beta * mem + input_ - reset
      return spk, mem

    @staticmethod
    class ATan(torch.autograd.Function):
       @staticmethod
       def forward(ctx, mem):
          spk = (mem > 0).float()
          ctx.save_for_backward(spk)
          return spk

       @staticmethod
       def backward(ctx, grad_output):
          (mem,) = ctx.saved_tensors
          grad = 1/(1+  (np.pi * mem).pow_(2)) * grad_output
          return grad
   
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
          cur1 = self.fc1(x) # post-synaptic current <-- spk_in x weight
          spk1, mem1 = self.lif1(cur1, mem1) # mem[t+1] <--post-syn current + decayed membrane
          cur2 = self.fc2(spk1)
          spk2, mem2 = self.lif2(cur2, mem2)

          mem2_rec.append(mem2)
          spk2_rec.append(spk2)

       return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(data.size(0), -1))
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
   
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

# network of SNN
beta = 0.8
input_layer = 784
hidden_layer = 1000
output_layer = 10

num_steps = 25

net = Net().to(device)


loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9,0.999))

data, targets = next(iter(train_loader))

data = data.to(device)
targets = targets.to(device)

spk_rec, mem_rec = net(data.view(batch_size, -1))

loss_val = torch.zeros((1), dtype=dtype, device=device)

for step in range(num_steps):
   loss_val += loss(mem_rec[step], targets)

print(f"Training loss: {loss_val.item():.3f}")

num_epochs = 1
loss_hist = []
test_loss_hist = []
counter = 0

# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        spk_rec, mem_rec = net(data.view(data.size(0), -1))

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = net(test_data.view(test_data.size(0), -1))

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 50 == 0:
                train_printer()
            counter += 1
            iter_counter +=1

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()


total = 0
correct = 0

with torch.no_grad():
  net.eval()
  for data, targets in test_loader:
    data = data.to(device)
    targets = targets.to(device)

    # forward pass
    test_spk, _ = net(data.view(data.size(0), -1))

    # calculate total accuracy
    _, predicted = test_spk.sum(dim=0).max(1)
    total += targets.size(0)
    correct += (predicted == targets).sum().item()

print(f"Number correct = {correct}/{total}")
print(f"Percentage correct = {(correct/total * 100):.2f}")