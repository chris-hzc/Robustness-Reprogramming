import math

import torch
from torch import nn


import math

import torch



class RobustConv2d(torch.nn.Module):
    """
    Custom implementation of 2D convolutional layer

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: size of the kernel (filter)
    :param stride: stride of the kernel (filter)
    :param padding: padding of the kernel (filter)
    :param dilation: dilation of the kernel (filter)
    :param groups: number of blocked connections from input channels to output channels
    :param bias: whether to use bias or not

    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = True):
        super(RobustConv2d, self).__init__()

        assert in_channels > 0, "in_channels must be greater than 0"
        assert out_channels > 0, "out_channels must be greater than 0"
        assert kernel_size > 0, "kernel_size must be greater than 0"
        assert stride > 0, "stride must be greater than 0"
        assert padding >= 0, "padding must be greater or equal to 0"
        assert dilation > 0, "dilation must be greater than 0"
        assert groups > 0, "groups must be greater than 0"

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cache = None

        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
            
        self.robust_sum = RobustSum(K=1, norm="L2", gamma=4.0, delta=3.0, N=1)
    

    def forward(self, input_data: torch.Tensor):
        """
        Forward pass of the layer

        :param input_data: input data of shape (batch_size, in_channels, height, width)
        :return: output data of shape (batch_size, out_channels, out_height, out_width)
        """
        assert len(input_data.shape) == 4, "Input data must have shape (batch_size, in_channels, height, width)"

        batch_size, in_channels, in_height, in_width = input_data.shape
        out_channels, _, kernel_height, kernel_width = self.weight.shape

        out_height = ((in_height + 2 * self.padding - self.dilation * (kernel_height - 1) - 1) // self.stride) + 1
        out_width = ((in_width + 2 * self.padding - self.dilation * (kernel_width - 1) - 1) // self.stride) + 1

        padded_input = torch.nn.functional.pad(input_data, (self.padding, self.padding, self.padding, self.padding))
        self.cache = (input_data, padded_input, batch_size, out_channels, out_height, out_width)
        output = torch.zeros((batch_size, out_channels, out_height, out_width))
        
        
        ###########################################
        input_unfold = torch.nn.functional.unfold(input_data, kernel_size=(kernel_height, kernel_width), padding=self.padding, stride=self.stride, dilation=self.dilation)
        weight_unfold= self.weight.view(self.weight.shape[0],-1)
        #output_unfold = torch.matmul(weight_unfold, input_unfold)
        
        output_unfold = self.robust_sum(weight_unfold, input_unfold)
        
        '''
        aa = []
        for i in range(input_data.shape[1]):
            ip = torch.nn.functional.unfold(input_data[:,i,:,:].unsqueeze(1), kernel_size=(kernel_height, kernel_width), padding=self.padding, stride=self.stride, dilation=self.dilation)
            w = self.weight[:,i,:,:].view(self.weight.shape[0],-1)
            aa.append(self.robust_sum(w, ip))
        output_unfold = sum(aa)
        '''

        output = output_unfold.view(batch_size, self.weight.shape[0], out_height, out_width)

        '''
        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        output[b, c_out, h_out, w_out] = self._conv_forward(h_out, w_out, c_out, b, padded_input)
        '''

        
        ###########################################
        

        return output
    
    
class RobustSum(nn.Module):
    def __init__(self, K=1, norm="L2", gamma=4.0, delta=3.0, N=1, e=1e-3, lmbd=1.0):
        super().__init__()
        self.K=K
        self.norm=norm
        self.gamma=gamma
        self.delta=delta
        self.epsilon = e
        self.N=N
        self.lmbd=lmbd
        #self.epsilon = torch.nn.Parameter(torch.tensor([e]), requires_grad=True)
        

    def median(self, k, x):
        # x : B * (C1*KK) * (HW), k: C2 * (C1*KK), 

        
        N = k.shape[1]
        
        
        z = torch.matmul(k,x)
        
        z0 = z
        
        
        if self.norm == 'L2':
            return z
        
        #z = torch.zeros(x.shape[0],k.shape[0],x.shape[2]).cuda()
        #z = torch.randn(x.shape[0],k.shape[0],x.shape[2]).cuda()
        
        x_k = x.unsqueeze(1) * k.unsqueeze(0).unsqueeze(-1)
        


        for _ in range(self.K):

            dist = torch.abs(x_k - z.unsqueeze(2)/N)
            
            if self.norm == "L2":
                w = torch.ones(dist.shape).cuda()

            elif self.norm in ['L1', "L12"]:
                #w = 1/(dist + torch.exp(self.epsilon))
                w = 1/(dist + self.epsilon)
                
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
                
            w_norm = torch.nn.functional.normalize(w,p=1,dim=2)
            
            z = N * (w_norm * x_k).sum(dim=2) 
            
            
            torch.cuda.empty_cache()
            
        if self.norm == "L12":
            return self.lmbd * z0 + (1-self.lmbd) * z
        else:
            return z
    
    def forward(self, k, x):
        outs = []
        b = k.shape[1]//self.N
        for i in range(self.N):
            outs.append(self.median(k[:,i*b:i*b+b],x[:,i*b:i*b+b,:]))
        return sum(outs)
        

'''
class RobustSum(nn.Module):
    def __init__(self, K=3, norm="L2", gamma=4.0, delta=3.0):
        super().__init__()
        self.K=K
        self.norm=norm
        self.gamma=gamma
        self.delta=delta
        self.epsilon = 1e-3


    def forward(self, k, x):
        #weighted_receptive_field = receptive_field * self.weight[c_out]
        #output = weighted_receptive_field.sum()
        # x : C * K * K, k: C * K * K, 
         
        x_k = x * k
        z = x_k.sum()
        N = x.numel() # N = C * K * K
        
        if self.norm == 'L2':
            return z

        for _ in range(self.K):

            dist = torch.abs(x_k - z/N)
            

            if self.norm == 'L1':
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
            
            z = N * (w * x_k).sum() / w.sum()  
            
            
            torch.cuda.empty_cache()
        return z
'''




class CustomConv2d(torch.nn.Module):
    """
    Custom implementation of 2D convolutional layer

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: size of the kernel (filter)
    :param stride: stride of the kernel (filter)
    :param padding: padding of the kernel (filter)
    :param dilation: dilation of the kernel (filter)
    :param groups: number of blocked connections from input channels to output channels
    :param bias: whether to use bias or not

    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = True):
        super(CustomConv2d, self).__init__()

        assert in_channels > 0, "in_channels must be greater than 0"
        assert out_channels > 0, "out_channels must be greater than 0"
        assert kernel_size > 0, "kernel_size must be greater than 0"
        assert stride > 0, "stride must be greater than 0"
        assert padding >= 0, "padding must be greater or equal to 0"
        assert dilation > 0, "dilation must be greater than 0"
        assert groups > 0, "groups must be greater than 0"

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cache = None

        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_data: torch.Tensor):
        """
        Forward pass of the layer

        :param input_data: input data of shape (batch_size, in_channels, height, width)
        :return: output data of shape (batch_size, out_channels, out_height, out_width)
        """
        assert len(input_data.shape) == 4, "Input data must have shape (batch_size, in_channels, height, width)"

        batch_size, in_channels, in_height, in_width = input_data.shape
        out_channels, _, kernel_height, kernel_width = self.weight.shape

        out_height = ((in_height + 2 * self.padding - self.dilation * (kernel_height - 1) - 1) // self.stride) + 1
        out_width = ((in_width + 2 * self.padding - self.dilation * (kernel_width - 1) - 1) // self.stride) + 1

        padded_input = torch.nn.functional.pad(input_data, (self.padding, self.padding, self.padding, self.padding))
        self.cache = (input_data, padded_input, batch_size, out_channels, out_height, out_width)
        output = torch.zeros((batch_size, out_channels, out_height, out_width))

        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        output[b, c_out, h_out, w_out] = self._conv_forward(h_out, w_out, c_out, b, padded_input)

        return output

    def _conv_forward(self, h_out, w_out, c_out, b, padded_input):
        h_start = h_out * self.stride
        h_end = h_start + self.weight.size(2) * self.dilation
        w_start = w_out * self.stride
        w_end = w_start + self.weight.size(3) * self.dilation

        receptive_field = padded_input[b, :, h_start:h_end:self.dilation, w_start:w_end:self.dilation]
        weighted_receptive_field = receptive_field * self.weight[c_out]
        output = weighted_receptive_field.sum()

        if self.bias is not None:
            output += self.bias[c_out]

        return output

    def backward(self, grad_output: torch.Tensor):
        """
        Backward pass of the layer

        :param grad_output: gradient of the loss with respect to the output of the layer
        :return: gradient of the loss with respect to the input of the layer
        """

        assert len(grad_output.shape) == 4,\
            "Grad output must have shape (batch_size, out_channels, out_height, out_width)"

        input_data, padded_input, batch_size, out_channels, out_height, out_width = self.cache
        grad_input = torch.zeros_like(padded_input, device=self.weight.device)
        grad_weight = torch.zeros_like(self.weight)
        grad_bias = torch.zeros_like(self.bias) if self.bias is not None else None

        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        h_end = h_start + self.weight.size(2) * self.dilation
                        w_start = w_out * self.stride
                        w_end = w_start + self.weight.size(3) * self.dilation

                        receptive_field = padded_input[b, :, h_start:h_end:self.dilation, w_start:w_end:self.dilation]

                        grad_input[b, :, h_start:h_end:self.dilation, w_start:w_end:self.dilation] += \
                            grad_output[b, c_out, h_out, w_out] * self.weight[c_out]

                        grad_weight[c_out] += grad_output[b, c_out, h_out, w_out] * receptive_field

                        if self.bias is not None:
                            grad_bias[c_out] += grad_output[b, c_out, h_out, w_out]

        self.weight.grad = grad_weight / batch_size
        if self.bias is not None:
            self.bias.grad = grad_bias / batch_size

        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

        assert (grad_input.shape == padded_input.shape),\
            f"Grad input shape ({grad_input.shape}) is not equal to input shape ({padded_input.shape})"

        return grad_input, grad_weight, grad_bias
