import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
import math


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        #self.conv1 = RobustConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = RobustConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RobustSum(nn.Module):
    def __init__(self, K=3, norm="L2", gamma=4.0, delta=3.0):
        super().__init__()
        self.K=K
        self.norm=norm
        self.gamma=gamma
        self.delta=delta
        self.epsilon = 1e-3


    def forward(self, x, weight):
        
        B = x.shape[0]
        D1 = weight.shape[0]
        
        z = torch.matmul(x, weight)
        
        if self.norm == 'L2':
            return z
        
        xw = x.unsqueeze(-1) * weight.unsqueeze(0)
        
        for _ in range(self.K):

            dist = torch.abs(xw - z.unsqueeze(1)/D1)
            
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
                
            w_norm = torch.nn.functional.normalize(w,p=1,dim=1)
            
            
            z = D1 * (w_norm * xw).sum(dim=1)
            
            
            torch.cuda.empty_cache()
        return z


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





    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 8

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.layer1 = self._make_layer(block, 8, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 16, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 32, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)
        self.linear = RobustLinear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        track = False
        
        if track:
            zs = [x] 
        out = F.relu(self.bn1(self.conv1(x)))
        if track:
            zs.append(out)
        out = self.layer1(out)
        if track:
            zs.append(out)
        out = self.layer2(out)
        if track:
            zs.append(out)
        out = self.layer3(out)
        if track:
            zs.append(out)
        out = self.layer4(out)
        if track:
            zs.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if track:
            zs.append(out)
        return out#, zs
    

    
    


def PreActTinyRobMLP18():
    return ResNet(PreActBlock, [2,2,2,2])

def TinyRobMLP10():
    return ResNet(BasicBlock, [1,1,1,1])

def TinyRobMLP18():
    return ResNet(BasicBlock, [2,2,2,2])

def TinyRobMLP34():
    return ResNet(BasicBlock, [3,4,6,3])

def TinyRobMLP50():
    return ResNet(Bottleneck, [3,4,6,3])

def TinyRobMLP101():
    return ResNet(Bottleneck, [3,4,23,3])

def TinyRobMLP152():
    return ResNet(Bottleneck, [3,8,36,3])
