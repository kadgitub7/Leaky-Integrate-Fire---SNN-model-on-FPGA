import math
import numpy
import matplotlib.pyplot as plt
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Value():
    def __init__(self, data, _children=(), _op = '', label=''):
        self.data = data
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op
        self.grad = 0
        self.label = label
    def __repr__(self):
        return f"Value = {self.data}"
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        self = self if isinstance(self, Value) else Value(self)
        sum = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += sum.grad * 1.0
            other.grad += sum.grad * 1.0
        sum._backward = _backward
        return sum
    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        product = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += product.grad * other.data
            other.grad += product.grad * self.data
        product._backward = _backward
        return product
    def __pow__(self, other):
        # Ensure the exponent is an int or float, not a Value object
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            # d/dx (x^n) = n * x^(n-1)
            self.grad += (other * (self.data ** (other - 1))) * out.grad
            
        out._backward = _backward
        return out
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other
    def tanh(self):
        e2x = math.exp(2* self.data)
        t = (e2x - 1)/(e2x + 1)
        o = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * o.grad
        o._backward = _backward
        return o
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    def __call__(self, x):
        act = sum((wi * xi for wi,xi in zip(self.w, x)), self.b)
        flatten_act = act.tanh()
        return flatten_act
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

batch_size = 128

data_path='/tmp/data/mnist'

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

net = MLP(784, [20, 20, 10])

for batch_idx, (images, labels) in enumerate(train_loader):
    batch_loss = Value(0.0)
    for i in range(images.size(0)):
        xs = images[i].numpy().flatten()
        neuron_inputs = [Value(float(px)) for px in xs]
        true_class = int(labels[i].item())
        pred_class = net(neuron_inputs)

        targets = [-1.0] * 10
        targets[true_class] = 1.0

        loss = sum((yp - yg)**2 for yp, yg in zip(pred_class, targets))
        batch_loss += loss

    print(f"batch_loss{batch_idx} = {batch_loss.data}")

    for p in net.parameters():
        p.grad = 0.0
    batch_loss.backward()
    
    for p in net.parameters():
        p.data -= 0.01 * p.grad

total = 0
correct = 0

net.eval()
for batch_idx, (images, labels) in enumerate(test_loader):
    for i in range(images.size(0)):
        xs = images[i].numpy().flatten()
        neuron_inputs = [Value(float(px)) for px in xs]
        true_class = int(labels[i].item())
        pred_class = net(neuron_inputs)

        prediction = pred_class.index(max(pred_class, key=lambda v: v.data))

        if prediction == true_class:
            correct += 1
        total += 1

print(f"Number correct = {correct}/{total}")
print(f"Percentage correct = {(correct/total * 100):.2f}")