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
import time
time_str = time.strftime('%Y-%m-%d-%H-%M')

randomSeed = 1
random.seed(randomSeed)  # python random seed
torch.manual_seed(randomSeed)  # pytorch random seed
np.random.seed(randomSeed)  # numpy random seed

path = ''


parser = argparse.ArgumentParser(description='MLP')

parser.add_argument('--norm',type=str, default='L21')
parser.add_argument('--K',type=int, default=1)
parser.add_argument('--e',type=float, default=5e-5)

parser.add_argument('--lmbd',type=float, default=0.6) # balance between L2 and L1

parser.add_argument('--lr',type=float, default=0.001)

parser.add_argument('--pretrain',type=int, default=1)

parser.add_argument('--num_linear',type=int, default=3) # 1:linear_784_10, 2: linear_784_64_10, 3:linear_784_256_64_10

parser.add_argument('--train_mode',type=int, default=0) # 1: training 0: testing
parser.add_argument('--adv_train',type=bool, default=False) # True: adv-train, False: normal-train

parser.add_argument('--attack',type=str, default='Linf')
parser.add_argument('--epsilon',type=float, default=0.3) # adv train with epsilon


parser.add_argument('--paradigm',type=int, default=1) # 1 only train lambda, 2: train all the model parammeters


args = parser.parse_args()
archs = {1:f"linear_784_10", 2: f"linear_784_64_10", 3:f"linear_784_256_64_10"}


arch = archs[args.num_linear]
pretrained_model = path + f"mlp3_mnist_model.pth"
# pretrained_model = path + f"mnist_saved_model/mlp_{arch}/K1_normL2_e0.001_advFalse_eps0.3_paradigm2/epoch9.pth"



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
    def __init__(self, K=3, norm="L2", gamma=4.0, delta=3.0, epsilon=1e-3, lmbd = 1.0):
        super().__init__()
        self.K=K
        self.norm=norm
        self.gamma=gamma
        self.delta=delta
        self.epsilon = epsilon
        # self.lmbd = lmbd
        self.lmbd = torch.nn.Parameter(torch.tensor([lmbd]), requires_grad=True)


    def forward(self, x, weight):
        
        D1 = weight.shape[0]
        
        z = torch.matmul(x, weight)
        
        z0 = z
        
        if self.norm == 'L2':
            return z
        
        xw = x.unsqueeze(1) * weight.T.unsqueeze(0)
        
        for _ in range(self.K):

            dist = torch.abs(xw - z.unsqueeze(-1)/D1)
            
            if self.norm == "L2":
                w = torch.ones(dist.shape).cuda()

            elif self.norm in  ['L1', "L21"]:
                w = 1/(dist+self.epsilon)
                
                
            w_norm = torch.nn.functional.normalize(w,p=1,dim=-1)
            
            
            z = D1 * (w_norm * xw).sum(dim=-1)
            
            
            torch.cuda.empty_cache()
        if self.norm == "L21":
            if args.paradigm == 1:
                return self.lmbd * z0 + (1-self.lmbd) * z
            elif args.paradigm == 2:
                return torch.sigmoid(self.lmbd) * z0 + (1-torch.sigmoid(self.lmbd)) * z
        else:
            return z

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if args.num_linear == 1:
            self.fc1 = RobustLinear(28*28, 10)
        elif args.num_linear == 2:
            self.fc1 = RobustLinear(28*28, 64)
            self.fc2 = RobustLinear(64, 10)
        elif args.num_linear == 3:
            self.fc1 = RobustLinear(28*28, 256)
            self.fc2 = RobustLinear(256, 64)
            self.fc3 = RobustLinear(64, 10)

    def forward(self, x):
        zs = [x]
        # Dimensions
        # 28, 28,  1
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        zs.append(x)
        if args.num_linear>1:
            x = self.fc2(x)
            zs.append(x)
        if args.num_linear>2:
            x = self.fc3(x)
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

test_loader = torch.utils.data.DataLoader(sampled_dataset, batch_size=1, shuffle=True)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
if args.pretrain:
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



def lnorm_attack(image, epsilon, data_grad):
    
    p = 1 if args.attack == "L1" else 2
    grad_norms = (
        torch.norm(data_grad.view(-1), p=p)
        + 1e-10
    )  # nopep8
    grad = data_grad / grad_norms
    adv_images = image.detach() + 0.3 * grad

    delta = adv_images - image
    delta_norms = torch.norm(delta.view(-1), p=p)
    factor = epsilon / delta_norms
    # factor = torch.min(factor, torch.ones_like(delta_norms))
    delta = delta * factor

    adv_images = torch.clamp(image + delta, min=0, max=1).detach()
    
    return adv_images

def l0_attack(image, epsilon, data_grad):
    

    topk_values, topk_indices = torch.topk(data_grad.view(-1), epsilon)

    mask = torch.zeros_like(image)


    mask.view(-1)[topk_indices] = 1

    grad = image * mask
    
    
    adv_images = image.detach() + 5.0 * grad

    delta = adv_images - image

    adv_images = torch.clamp(image + delta, min=0, max=1).detach()
    
    return adv_images

def test( model, device, test_loader, epsilon ):
    correct = 0
    adv_examples = []
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
        elif args.attack in ["L2","L1"]:
            perturbed_data = lnorm_attack(data, epsilon, data_grad)
        elif args.attack == 'L0':
            perturbed_data = l0_attack(data, epsilon, data_grad)
            

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

    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    print("lmbd:",  [torch.sigmoid(module.robust_sum.lmbd).item() for module in model.modules() if isinstance(module, RobustLinear)])

    return final_acc, adv_examples

accuracies = []
examples = []


if args.attack == "Linf":
    epsilons = [0, .05, .1, .15, .2, .25, .3]
elif args.attack == 'L2':
    epsilons = [0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
elif args.attack == "L1":
    epsilons = [0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
elif args.attack == "L0":
    epsilons = [0, 128, 256, 512]
    
K, norm = args.K, args.norm
print("K, norm, lmbd: ", K, norm, args.lmbd)
def load_params(model):
    i=0
    for module in model.modules():
        if isinstance(module, RobustLinear):
            module.robust_sum.K = K
            module.robust_sum.norm = norm
            module.robust_sum.epsilon = args.e
            # module.robust_sum.lmbd = args.lmbd
            
            module.robust_sum.lmbd.data.fill_(args.lmbd)
            
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
    #print(accuracies)
    print(" & ".join([str(x*100) for x in accuracies]))


    

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


    if args.paradigm == 1:
        for param in model.parameters():
            param.requires_grad = False
        
        for module in model.modules():
            #if isinstance(module, RobustConv2d):
            if isinstance(module, RobustLinear):
                module.robust_sum.lmbd.requires_grad = True
            
    
    

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Training settings
    epochs = 10
    epsilon = args.epsilon  # Perturbation parameter for FGSM
    

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
                
                print("lmbd:",  [torch.sigmoid(module.robust_sum.lmbd).item() for module in model.modules() if isinstance(module, RobustLinear)])
            # print(list(model.parameters())[-2])
        model_param = f"K{K}_norm{norm}_e{args.e}"
        os.makedirs(path+f'mnist_saved_model/mlp_{arch}/{model_param}_adv{adv}_eps{epsilon}_paradigm{args.paradigm}/', exist_ok=True)
        
        torch.save(model.state_dict(), path+f'mnist_saved_model/mlp_{arch}/{model_param}_adv{adv}_eps{epsilon}_paradigm{args.paradigm}/epoch{epoch}.pth')
        model.eval()
        acc, _ = test(model, device, test_loader, 0)
        adv_acc, _ = test(model, device, test_loader, epsilon)
        accs.append(acc)
        adv_accs.append(adv_acc)
        #print(f"Epoch {epoch}:")
        #print(f"acc: {acc}, adv acc: {adv_acc}")
        # print("lmbd:",  [torch.sigmoid(module.robust_sum.lmbd).item() for module in model.modules() if isinstance(module, RobustLinear)])
        
    print("accs:", accs)
    print("adv_accs:", adv_accs)
    print("lmbd:",  [torch.sigmoid(module.robust_sum.lmbd).item() for module in model.modules() if isinstance(module, RobustLinear)])
    print(list(model.parameters())[-2])

    # return model
    return accs, adv_accs





if __name__ == '__main__':
    


    
    os.makedirs(path+f'mnist_result/train{args.train_mode}_paradigm{args.paradigm}/mlp_{arch}/', exist_ok=True)
    sys.stdout = open(path+f'mnist_result/train{args.train_mode}_paradigm{args.paradigm}/mlp_{arch}/K{args.K}_norm{args.norm}_e{args.e}_adv{args.adv_train}_eps{args.epsilon}_{time_str}.log', 'w', buffering=1)
    if args.train_mode:
        accs, adv_accs = adv_train(adv=args.adv_train)
    else:
        test_all() # test plug in
    sys.stdout.close()
    
    



