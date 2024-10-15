from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter

from models.conv2d import RobustConv2d
import math

import random
import numpy as np
import sys
import os
import argparse
randomSeed = 1
random.seed(randomSeed)  # python random seed
torch.manual_seed(randomSeed)  # pytorch random seed
np.random.seed(randomSeed)  # numpy random seed

path = ''


parser = argparse.ArgumentParser(description='LeNet')

parser.add_argument('--norm',type=str, default='L12')
parser.add_argument('--gamma',type=float, default=2.0)
parser.add_argument('--delta',type=float, default=1.0)
parser.add_argument('--K',type=int, default=1)
parser.add_argument('--e',type=float, default=1e-3)

parser.add_argument('--lmbd',type=float, default=0.3)

parser.add_argument('--attack',type=str, default='Linf')

args = parser.parse_args()

pretrained_model = path + "lenet_mnist_model.pth"


use_cuda=True


class RobustLinear(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.robust_sum = RobustSum(K=3, norm="L2", gamma=4.0, delta=3.0)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # return F.linear(input, self.weight, self.bias)
        y =  self.robust_sum(input, self.weight.T)
        return y + self.bias

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class RobustSum(nn.Module):
    def __init__(self, K=3, norm="L2", gamma=4.0, delta=3.0, epsilon=1e-3):
        super().__init__()
        self.K=K
        self.norm=norm
        self.gamma=gamma
        self.delta=delta
        self.epsilon = epsilon


    def forward(self, x, weight):
        
        D1 = weight.shape[0]
        
        z = torch.matmul(x, weight)
        
        if self.norm == 'L2':
            return z
        
        xw = x.view(-1,1) * weight
        
        for _ in range(self.K):

            dist = torch.abs(xw - z/D1)
            
            if self.norm == "L2":
                w = torch.ones(dist.shape).cuda()

            elif self.norm == 'L1':
                w = 1/(dist+self.epsilon)
                
            elif  self.norm == 'MCP':
                w = 1/(dist + self.epsilon) - 1/self.gamma
                w[w<self.epsilon]=self.epsilon
                
            elif self.norm == 'Huber':
                w = self.delta/(dist + self.epsilon)
                w[w>1.0] = 1.0
            elif self.norm == 'HM':
                w = self.delta/(self.gamma-self.delta)*(self.gamma/(dist + self.epsilon)-1.0)
                w[w>1.0] = 1.0
                w[w<self.epsilon]=self.epsilon
                
            w_norm = torch.nn.functional.normalize(w,p=1,dim=0)
            
            
            z = D1 * (w_norm * xw).sum(dim=0).view(1,-1) 
            
            
            torch.cuda.empty_cache()
        return z

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = RobustConv2d(1, 10, kernel_size=5)
        self.conv2 = RobustConv2d(10, 20, kernel_size=5)
        
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = RobustLinear(320, 50)
        self.fc2 = RobustLinear(50, 10)

    def forward(self, x):
        zs = [x]
        # Dimensions
        # 28, 28,  1
        x = self.conv1(x)
        zs.append(x)
        x = F.relu(F.max_pool2d(x, 2))
        
        # 12, 12, 10
        x = self.conv2(x)
        zs.append(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(x), 2))
        #  4,  4, 20
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        zs.append(x)
        x = self.fc2(x)
        zs.append(x)
        return F.log_softmax(x, dim=1),zs
    



    
# MNIST Test dataset and dataloader declaration
mnist_dataset = datasets.MNIST(
    path+'data',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ]),
)

sampled_dataset = torch.utils.data.Subset(mnist_dataset, range(1000))

test_loader = torch.utils.data.DataLoader(sampled_dataset, batch_size=1, shuffle=False)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def l2_attack(image, epsilon, data_grad):

    grad_norms = (
        torch.norm(grad.view(-1), p=2, dim=1)
        + 1e-10
    )  # nopep8
    grad = grad / grad_norms
    adv_images = adv_images.detach() + 0.3 * grad

    delta = adv_images - image
    delta_norms = torch.norm(delta.view(-1), p=2, dim=1)
    factor = epsilon / delta_norms
    factor = torch.min(factor, torch.ones_like(delta_norms))
    delta = delta * factor

    adv_images = torch.clamp(image + delta, min=0, max=1).detach()
    
    
    return adv_images


def test( model, device, test_loader, epsilon ):
    correct = 0
    adv_examples = []
    labels = []
    i=0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output, zs = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()

        data_grad = data.grad.data
        
        if args.attack == "Linf":
            perturbed_data = fgsm_attack(data, epsilon, data_grad) 
        elif args.attack == "L2":
            perturbed_data = l2_attack(data, epsilon, data_grad) 

        # Re-classify the perturbed image
        output, zs_adv = model(perturbed_data)
        

        
        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            
            
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        i+=1
        # print(i)
    print(labels)
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    return final_acc, adv_examples

accuracies = []
examples = []



epsilons = [0, .05, .1, .15, .2, .25, .3]

K, norm, gamma, delta = args.K, args.norm, args.gamma, args.delta
print("K, norm, lmbd: ", K, norm, args.lmbd)
def load_params(model):
    i=0
    for module in model.modules():
        if isinstance(module, RobustConv2d):
        #if isinstance(module, RobustLinear):
            module.robust_sum.K = K
            module.robust_sum.norm = norm
            module.robust_sum.gamma = gamma
            module.robust_sum.delta = delta
            module.robust_sum.epsilon = args.e
            module.robust_sum.lmbd = args.lmbd
            
            i+=1
    print(f"Replace {i} layers.")
    return model

model = load_params(model)


def test_all():
    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)
        
    print(epsilons)
    # print(accuracies)
    print(" & ".join([f"{str(x*100)}" for x in accuracies]))



def test_for_epochs():
    accs, adv_accs = [], []
    for epoch in range(10):
        print(f"Epoch {epoch}:")
        model.eval()
        model_param = f"K{K}_norm{norm}_gamma{gamma}_delta{delta}"
        model.load_state_dict(torch.load(path+f'mnist/{model_param}/epoch{epoch}.pth'))
        acc, _ = test(model, device, test_loader, 0)
        adv_acc, _ = test(model, device, test_loader, 0.3)
        accs.append(acc)
        adv_accs.append(adv_acc)

    print("accs:", accs)
    print("adv_accs:", adv_accs)


    

def adv_train(adv=True):
    mnist_trainset = datasets.MNIST(
    path + 'data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
    )
    
    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Training settings
    epochs = 10
    epsilon = 0.3  # Perturbation parameter for FGSM
    

    accs = []
    adv_accs = []
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            data.requires_grad = True
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(data)
            loss = F.nll_loss(output, target)
            
            # Backward pass
            loss.backward()
            
            # Generate adversarial data
            data_grad = data.grad.data
            if adv:
                perturbed_data = fgsm_attack(data, epsilon, data_grad)
            else:
                perturbed_data = data
            
            # Re-classify the perturbed image
            output, _ = model(perturbed_data)
            
            # Calculate loss
            loss = F.nll_loss(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                
        model_param = f"K{K}_norm{norm}_gamma{gamma}_delta{delta}"
        os.makedirs(path+f'mnist_lenet/{model_param}_adv{adv}/', exist_ok=True)
        
        torch.save(model.state_dict(), path+f'mnist_lenet/{model_param}_adv{adv}/epoch{epoch}.pth')
        model.eval()
        acc, _ = test(model, device, test_loader, 0)
        adv_acc, _ = test(model, device, test_loader, epsilon)
        accs.append(acc)
        adv_accs.append(adv_acc)
        #print(f"Epoch {epoch}:")
        #print(f"acc: {acc}, adv acc: {adv_acc}")
    print("accs:", accs)
    print("adv_accs:", adv_accs)
    return accs, adv_accs





if __name__ == '__main__':
    
    test_all() # test plug in
    # accs, adv_accs = adv_train(adv=True)
    
    



