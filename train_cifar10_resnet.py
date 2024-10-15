import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os

import argparse
from models import *
from models.conv2d_learn import RobustLearnConv2d

import time
#from utils import progress_bar
import random
import numpy as np
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

randomSeed = 1
random.seed(randomSeed)  # python random seed
torch.manual_seed(randomSeed)  # pytorch random seed
np.random.seed(randomSeed)  # numpy random seed

time_str = time.strftime('%Y-%m-%d-%H-%M')

path = ''

parser = argparse.ArgumentParser(description='ResNet')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs',type=int, default=100)

# attack
parser.add_argument('--budget', default=8, type=int)
parser.add_argument('--alpha', default=2, type=int)
parser.add_argument('--k', default=7, type=int)

# model

parser.add_argument('--norm',type=str, default='L12')
parser.add_argument('--beta',type=float, default=0.0)


parser.add_argument('--K',type=int, default=1)
parser.add_argument('--tiny',type=int, default=1)
parser.add_argument('--N',type=int, default=1)
parser.add_argument('--e',type=float, default=1e-3)
parser.add_argument('--start_robust',type=int, default=0)
parser.add_argument('--pretrain',type=int, default=1)
parser.add_argument('--paradigm',type=int, default=1) # 1 only train lambda, 2: train all the model parammeters

parser.add_argument('--resnet',type=int, default=18)

parser.add_argument('--batch_train',type=int, default=128)
parser.add_argument('--batch_test',type=int, default=100)





args = parser.parse_args()

    
if args.tiny:
    arch = f"tiny_resnet{args.resnet}_norm{args.norm}_beta{args.beta}_K{args.K}_e{args.e}_N{args.N}_start{args.start_robust}_iter{args.k}_paradigm{args.paradigm}"
else:
    arch = f"resnet{args.resnet}_norm{args.norm}_K{args.K}_e{args.e}_start{args.start_robust}"



#args.e =  math.log(args.e)

def load_params(args, model):
    i=0
    for module in model.modules():
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
    print(f"Replace {i} layers.")
    return model



learning_rate = args.lr
epsilon = args.budget/255
k = args.k
alpha = args.alpha/255

#file_name = 'pgd_adv_train_adaptive'
file_name = 'pgd_adv_train_learn'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root=path+'data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root=path+'data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_train, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_test, shuffle=False, num_workers=4)

class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv

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

if args.pretrain:
        
    checkpoint = torch.load(path + f'tinyresnet18_cifar10_model.t7' )
    net.load_state_dict(checkpoint['net'])
    
net = load_params(args, net)


adversary = LinfPGDAttack(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

def train(epoch):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    
    if args.paradigm == 1:
        for param in net.parameters():
            param.requires_grad = False
        
        for module in net.modules():
            #if isinstance(module, RobustConv2d):
            if isinstance(module, RobustLearnConv2d):
                module.robust_sum.beta.requires_grad = True
                
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        adv = adversary.perturb(inputs, targets)
        adv_outputs = net(adv)
        loss = criterion(adv_outputs, targets)
        loss.backward()

        betas = []
        for module in net.modules():
            if isinstance(module, RobustLearnConv2d):
                betas.append(torch.sigmoid(module.robust_sum.beta.data).item())
        print(betas)

        optimizer.step()
        train_loss += loss.item()
        _, predicted = adv_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current adversarial train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current adversarial train loss:', loss.item())
            print('\nTotal adversarial train accuarcy:', 100. * correct / total)
            print('Total adversarial train loss:', train_loss)
            print("beta:",  [torch.sigmoid(module.robust_sum.beta).item() for module in net.modules() if isinstance(module, RobustLearnConv2d)])

            print('\nTotal adversarial train accuarcy:', 100. * correct / total)
            print('Total adversarial train loss:', train_loss)
    print('\nTotal adversarial train accuarcy:', 100. * correct / total)
    print('Total adversarial train loss:', train_loss)
    
    return 100. * correct / total

def test(epoch):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            benign_loss += loss.item()

            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print('\nCurrent batch:', str(batch_idx))
                print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current benign test loss:', loss.item())

            adv = adversary.perturb(inputs, targets)
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
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)

    state = {'net': net.state_dict(),'acc': 100. * benign_correct / total, 'adv_acc': 100. * adv_correct / total, 'epoch': epoch}

    os.makedirs(path+f'checkpoint/{file_name}/{arch}', exist_ok=True)
    
    torch.save(state, path+f'checkpoint/{file_name}/{arch}/epoch{epoch}.t7')
    print('Model Saved!')
    return 100. * benign_correct / total, 100. * adv_correct / total


def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    
    if args.resnet > 10:
        if epoch >= 50:
            lr /= 10
        if epoch >= 75:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    
def main():
    print(args)
    train_accs, test_accs, test_acc_advs = [], [], []
    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train_acc = train(epoch)
        test_acc, test_acc_adv = test(epoch)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        test_acc_advs.append(test_acc_adv)
    print("train accs")
    print(train_accs)
    print("test accs")
    print(test_accs)
    print("test adv accs")
    print(test_acc_advs)

if __name__ == '__main__':
    os.makedirs(path+f'log/{file_name}', exist_ok=True)
    sys.stdout = open(path+f'log/{file_name}/{arch}_{time_str}.log', 'w', buffering=1)
    main()
    sys.stdout.close()