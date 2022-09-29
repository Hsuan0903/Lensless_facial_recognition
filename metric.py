import math
import torch.nn.functional as F
from torch.nn import Parameter
from torch import nn
import torch

class Softmax(nn.Module): # Just a Linear layer(weight layer) because the ``torch.nn.CrossEntropyLoss`` includes the softmax function and negative log-likelihood loss (torch.nn.NLLLoss) .
    def __init__(self,num_classes=10,embedding_size=512):
        super(Softmax, self).__init__()
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size),requires_grad=True) 
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        
        output = F.linear(input,self.weight)
        return output
   
class ArcFace(nn.Module):
    def __init__(self, num_classes=10,embedding_size=512,margin_m=0.5,margin_s=64.0):
        super(ArcFace, self).__init__()
        
        self.m = margin_m
        self.s = margin_s
                
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size),requires_grad=True)
        nn.init.xavier_uniform_(self.weight)
        
        
    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.weight)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output
    
class AdaCos(nn.Module):
    def __init__(self, num_classes=10, embedding_size=512):
        super(AdaCos, self).__init__()
        
        self.s = math.sqrt(2) * math.log(num_classes - 1)

        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size),requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.weight)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits
        
        return output

class SphereFace(nn.Module):
    def __init__(self, num_classes=10, embedding_size=512, margin_m=1.35, margin_s=64.0):
        super(SphereFace, self).__init__()
        self.embedding_size = embedding_size
        self.n_classes = num_classes
        self.s = margin_s
        self.m = margin_m
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size),requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.weight)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(self.m * theta)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output


class CosFace(nn.Module):
    def __init__(self,  num_classes=10, embedding_size=512, margin_m=0.35, margin_s=64.0):
        super(CosFace, self).__init__()
        self.num_features = embedding_size
        self.n_classes = num_classes
        self.s = margin_s
        self.m = margin_m
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size),requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.weight)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output
