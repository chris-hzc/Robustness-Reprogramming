import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
#from abc import ABC, abstractmethod

from models import *
#from advertorch.attacks import LinfPGDAttack
from autoattack import AutoAttack
from torch.utils.data import Subset

import os
import argparse
import torchattacks

import time
#from utils import progress_bar
import random
import numpy as np
import sys
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"


randomSeed = 1
random.seed(randomSeed)  # python random seed
torch.manual_seed(randomSeed)  # pytorch random seed
np.random.seed(randomSeed)  # numpy random seed

time_str = time.strftime('%Y-%m-%d-%H-%M')

path = ''

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

# attack
parser.add_argument('--budget', default=8, type=int)
parser.add_argument('--alpha', default=2, type=int)
parser.add_argument('--k', default=20, type=int)
parser.add_argument('--attack',type=str, default='PGD') #PGDL2, PGD, FGSM, Auto

# model
parser.add_argument('--norm',type=str, default='L12')

parser.add_argument('--beta',type=float, default=1.0)

parser.add_argument('--K',type=int, default=1)
parser.add_argument('--N',type=int, default=1)
parser.add_argument('--e',type=float, default=5e-4)

parser.add_argument('--batch',type=int, default=100)

parser.add_argument('--tiny',type=int, default=1)
parser.add_argument('--start_robust',type=int, default=0)
parser.add_argument('--epoch',type=int, default=75)

parser.add_argument('--resnet',type=int, default=18)



args = parser.parse_args()

if args.tiny:
    arch = f"tiny_resnet{args.resnet}_norm{args.norm}_beta{args.beta}_e{args.e}_epoch{args.epoch}"
else:
    arch = f"resnet{args.resnet}_norm{args.norm}_gamma{args.gamma}_delta{args.delta}_K{args.K}_e{args.e}_start{args.start_robust}"



def load_params(args, model):
    i=0
    for module in model.modules():
        #if isinstance(module, RobustConv2d):
        if isinstance(module, RobustLearnConv2d):
            if i >= args.start_robust:
                module.robust_sum.K = args.K
                module.robust_sum.norm = args.norm
                
                #module.robust_sum.beta = args.beta
                module.robust_sum.beta.data.fill_(args.beta)
                
                module.robust_sum.N = args.N
                module.robust_sum.epsilon = args.e
                print(f"Replace the {i}th layer!")
            i+=1
    return model



epsilon = args.budget/255
k = args.k
alpha = args.alpha/255


file_name = 'pgd_adv_train_learn'


device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = torchvision.datasets.CIFAR10(root=path + 'data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=4)

#net = ResNet18()



    
    
if args.tiny:
    if args.resnet == 10:
        net = TinyRobNetLearn10()
    elif args.resnet == 18:
        net = TinyRobNetLearn18()
    elif args.resnet == 34:
        net = TinyRobNetLearn34()
else:
    if args.resnet == 10:
        net = RobNetLearn10()
    elif args.resnet == 18:
        net = RobNetLearn18()
    elif args.resnet == 34:
        net = RobNetLearn34()
    
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True


if args.tiny:
    arch_load = "tiny_resnet18_normL2_gamma1.0_delta0.5_beta0.0_K1_e0.001_N1_start0_iter7"
else:
    arch_load = f"resnet{args.resnet}_norm{args.norm}_gamma{args.gamma}_delta{args.delta}_K{args.K}_e0.001_start{args.start_robust}"



checkpoint = torch.load(path + f'tinyresnet18_cifar10_model.t7' )
net.load_state_dict(checkpoint['net'])

net = load_params(args, net)

class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
                
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                #logits,_ = self.model(x)
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x
    
    
class FGSM(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)

        x.requires_grad_()
        with torch.enable_grad():
            logits,_ = self.model(x)
            #logits = self.model(x)
            loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + epsilon * torch.sign(grad.detach())
        x = torch.clamp(x, 0, 1)
        return x



if args.attack == 'PGD':
    adversary = LinfPGDAttack(net)
elif args.attack == 'FGSM':
    adversary = FGSM(net)
elif args.attack == 'PGDL2':
    adversary = torchattacks.PGDL2(net, eps=0.1, alpha=0.3, steps=7, random_start=True)
elif args.attack == 'AA':
    adversary = AutoAttack(net, norm='Linf', eps=epsilon)
    
    

criterion = nn.CrossEntropyLoss()

def test():
    print('\n[ Test Start ]')
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        outputs = net(inputs)
        #outputs = net(inputs)
        loss = criterion(outputs, targets)
        benign_loss += loss.item()

        _, predicted = outputs.max(1)
        benign_correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current benign test loss:', loss.item())
        if args.attack in ['PGDL2']:
            adv = adversary(inputs, targets)
        elif args.attack in ['AA']:
            adv = adversary.run_standard_evaluation(inputs, targets, bs=args.batch)
        else:
            adv = adversary.perturb(inputs, targets)
        #adv_outputs,zs_adv = net(adv)
        adv_outputs = net(adv)
        loss = criterion(adv_outputs, targets)
        adv_loss += loss.item()

        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current adversarial test loss:', loss.item())
            print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
            print('Total adversarial test Accuarcy:', 100. * adv_correct / total)

    print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
    print('Total adversarial test Accuarcy:', 100. * adv_correct / total)
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)
    
def test_aa():
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    num_exp = 1000
    adv = adversary.run_standard_evaluation(x_test[:num_exp], y_test[:num_exp],
                    bs=args.batch)
    
    
    
    

if __name__ == '__main__':
    os.makedirs(path+f'log_test/{file_name}', exist_ok=True)
    sys.stdout = open(path+f'log_test/{file_name}/{args.attack}_{arch}_K{args.K}_budget{args.budget}_iter{args.k}_{time_str}.log', 'w', buffering=1)
    if args.attack == 'AA':
        test_aa()
    else:
        test()
    sys.stdout.close()


