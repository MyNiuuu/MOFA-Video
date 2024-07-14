import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math

def MultiChannelSoftBinaryCrossEntropy(input, target, reduction='mean'):
    '''
    input: N x 38 x H x W --> 19N x 2 x H x W
    target: N x 19 x H x W --> 19N x 1 x H x W
    '''
    input = input.view(-1, 2, input.size(2), input.size(3))
    target = target.view(-1, 1, input.size(2), input.size(3))

    logsoftmax = nn.LogSoftmax(dim=1)
    if reduction == 'mean':
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

class EdgeAwareLoss():
    def __init__(self, nc=2, loss_type="L1", reduction='mean'):
        assert loss_type in ['L1', 'BCE'], "Undefined loss type: {}".format(loss_type)
        self.nc = nc
        self.loss_type = loss_type
        self.kernelx = Variable(torch.Tensor([[1,0,-1],[2,0,-2],[1,0,-1]]).cuda())
        self.kernelx = self.kernelx.repeat(nc,1,1,1)
        self.kernely = Variable(torch.Tensor([[1,2,1],[0,0,0],[-1,-2,-1]]).cuda())
        self.kernely = self.kernely.repeat(nc,1,1,1)
        self.bias = Variable(torch.zeros(nc).cuda())
        self.reduction = reduction
        if loss_type == 'L1':
            self.loss = nn.SmoothL1Loss(reduction=reduction)
        elif loss_type == 'BCE':
            self.loss = self.bce2d

    def bce2d(self, input, target):
        assert not target.requires_grad
        beta = 1 - torch.mean(target)
        weights = 1 - beta + (2 * beta - 1)  * target
        loss = nn.functional.binary_cross_entropy(input, target, weights, reduction=self.reduction)
        return loss

    def get_edge(self, var):
        assert var.size(1) == self.nc, \
            "input size at dim 1 should be consistent with nc, {} vs {}".format(var.size(1), self.nc)
        outputx = nn.functional.conv2d(var, self.kernelx, bias=self.bias, padding=1, groups=self.nc)
        outputy = nn.functional.conv2d(var, self.kernely, bias=self.bias, padding=1, groups=self.nc)
        eps=1e-05
        return torch.sqrt(outputx.pow(2) + outputy.pow(2) + eps).mean(dim=1, keepdim=True)

    def __call__(self, input, target):
        size = target.shape[2:4]
        input = nn.functional.interpolate(input, size=size, mode="bilinear", align_corners=True)
        target_edge = self.get_edge(target)
        if self.loss_type == 'L1':
            return self.loss(self.get_edge(input), target_edge)
        elif self.loss_type == 'BCE':
            raise NotImplemented
            #target_edge = torch.sign(target_edge - 0.1)
            #pred = self.get_edge(nn.functional.sigmoid(input))
            #return self.loss(pred, target_edge)

def KLD(mean, logvar):
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

class DiscreteLoss(nn.Module):
    def __init__(self, nbins, fmax):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        assert nbins % 2 == 1, "nbins should be odd"
        self.nbins = nbins
        self.fmax = fmax
        self.step = 2 * fmax / float(nbins)

    def tobin(self, target):
        target = torch.clamp(target, -self.fmax + 1e-3, self.fmax - 1e-3)
        quantized_target = torch.floor((target + self.fmax) / self.step)
        return quantized_target.type(torch.cuda.LongTensor)

    def __call__(self, input, target):
        size = target.shape[2:4]
        if input.shape[2] != size[0] or input.shape[3] != size[1]:
            input = nn.functional.interpolate(input, size=size, mode="bilinear", align_corners=True)
        target = self.tobin(target)
        assert input.size(1) == self.nbins * 2
        # print(target.shape)
        # print(input.shape)
        # print(torch.max(target))
        target[target>=99]=98  # odd bugs of the training loss. We have [0 ~ 99] in GT flow, but nbins = 99
        return self.loss(input[:,:self.nbins,...], target[:,0,...]) + self.loss(input[:,self.nbins:,...], target[:,1,...])

class MultiDiscreteLoss():
    def __init__(self, nbins=19, fmax=47.5, reduction='mean', xy_weight=(1., 1.), quantize_strategy='linear'):
        self.loss = nn.CrossEntropyLoss(reduction=reduction)
        assert nbins % 2 == 1, "nbins should be odd"
        self.nbins = nbins
        self.fmax = fmax
        self.step = 2 * fmax / float(nbins)
        self.x_weight, self.y_weight = xy_weight
        self.quantize_strategy = quantize_strategy

    def tobin(self, target):
        target = torch.clamp(target, -self.fmax + 1e-3, self.fmax - 1e-3)
        if self.quantize_strategy == "linear":
            quantized_target = torch.floor((target + self.fmax) / self.step)
        elif self.quantize_strategy == "quadratic":
            ind = target.data > 0
            quantized_target = target.clone()
            quantized_target[ind] = torch.floor(self.nbins * torch.sqrt(target[ind] / (4 * self.fmax)) + self.nbins / 2.)
            quantized_target[~ind] = torch.floor(-self.nbins * torch.sqrt(-target[~ind] / (4 * self.fmax)) + self.nbins / 2.)
        return quantized_target.type(torch.cuda.LongTensor)

    def __call__(self, input, target):
        size = target.shape[2:4]
        target = self.tobin(target)
        if isinstance(input, list):
            input = [nn.functional.interpolate(ip, size=size, mode="bilinear", align_corners=True) for ip in input]
            return sum([self.x_weight * self.loss(input[k][:,:self.nbins,...], target[:,0,...]) + self.y_weight * self.loss(input[k][:,self.nbins:,...], target[:,1,...]) for k in range(len(input))]) / float(len(input))
        else:
            input = nn.functional.interpolate(input, size=size, mode="bilinear", align_corners=True)
            return self.x_weight * self.loss(input[:,:self.nbins,...], target[:,0,...]) + self.y_weight * self.loss(input[:,self.nbins:,...], target[:,1,...])

class MultiL1Loss():
    def __init__(self, reduction='mean'):
        self.loss = nn.SmoothL1Loss(reduction=reduction)

    def __call__(self, input, target):
        size = target.shape[2:4]
        if isinstance(input, list):
            input = [nn.functional.interpolate(ip, size=size, mode="bilinear", align_corners=True) for ip in input]
            return sum([self.loss(input[k], target) for k in range(len(input))]) / float(len(input))
        else:
            input = nn.functional.interpolate(input, size=size, mode="bilinear", align_corners=True)
            return self.loss(input, target)

class MultiMSELoss():
    def __init__(self):
        self.loss = nn.MSELoss()
    
    def __call__(self, predicts, targets):
        loss = 0
        for predict, target in zip(predicts, targets):
            loss += self.loss(predict, target)
        return loss
        
class JointDiscreteLoss():
    def __init__(self, nbins=19, fmax=47.5, reduction='mean', quantize_strategy='linear'):
        self.loss = nn.CrossEntropyLoss(reduction=reduction)
        assert nbins % 2 == 1, "nbins should be odd"
        self.nbins = nbins
        self.fmax = fmax
        self.step = 2 * fmax / float(nbins)
        self.quantize_strategy = quantize_strategy
        
    def tobin(self, target):
        target = torch.clamp(target, -self.fmax + 1e-3, self.fmax - 1e-3)
        if self.quantize_strategy == "linear":
            quantized_target = torch.floor((target + self.fmax) / self.step)
        elif self.quantize_strategy == "quadratic":
            ind = target.data > 0
            quantized_target = target.clone()
            quantized_target[ind] = torch.floor(self.nbins * torch.sqrt(target[ind] / (4 * self.fmax)) + self.nbins / 2.)
            quantized_target[~ind] = torch.floor(-self.nbins * torch.sqrt(-target[~ind] / (4 * self.fmax)) + self.nbins / 2.)
        else:
            raise Exception("No such quantize strategy: {}".format(self.quantize_strategy))
        joint_target = quantized_target[:,0,:,:] * self.nbins + quantized_target[:,1,:,:]
        return joint_target.type(torch.cuda.LongTensor)

    def __call__(self, input, target):
        target = self.tobin(target)
        assert input.size(1) == self.nbins ** 2
        return self.loss(input, target)

class PolarDiscreteLoss():
    def __init__(self, abins=30, rbins=20, fmax=50., reduction='mean', ar_weight=(1., 1.), quantize_strategy='linear'):
        self.loss = nn.CrossEntropyLoss(reduction=reduction)
        self.fmax = fmax
        self.rbins = rbins
        self.abins = abins
        self.a_weight, self.r_weight = ar_weight
        self.quantize_strategy = quantize_strategy

    def tobin(self, target):
        indxneg = target.data[:,0,:,:] < 0
        eps = torch.zeros(target.data[:,0,:,:].size()).cuda()
        epsind = target.data[:,0,:,:] == 0
        eps[epsind] += 1e-5
        angle = torch.atan(target.data[:,1,:,:] / (target.data[:,0,:,:] + eps))
        angle[indxneg] += np.pi
        angle += np.pi / 2 # 0 to 2pi
        angle = torch.clamp(angle, 0, 2 * np.pi - 1e-3)
        radius = torch.sqrt(target.data[:,0,:,:] ** 2 + target.data[:,1,:,:] ** 2)
        radius = torch.clamp(radius, 0, self.fmax - 1e-3)
        quantized_angle = torch.floor(self.abins * angle / (2 * np.pi))
        if self.quantize_strategy == 'linear':
            quantized_radius = torch.floor(self.rbins * radius / self.fmax)
        elif self.quantize_strategy == 'quadratic':
            quantized_radius = torch.floor(self.rbins * torch.sqrt(radius / self.fmax))
        else:
            raise Exception("No such quantize strategy: {}".format(self.quantize_strategy))
        quantized_target = torch.autograd.Variable(torch.cat([torch.unsqueeze(quantized_angle, 1), torch.unsqueeze(quantized_radius, 1)], dim=1))
        return quantized_target.type(torch.cuda.LongTensor)

    def __call__(self, input, target):
        target = self.tobin(target)
        assert (target >= 0).all() and (target[:,0,:,:] < self.abins).all() and (target[:,1,:,:] < self.rbins).all()
        return self.a_weight * self.loss(input[:,:self.abins,...], target[:,0,...]) + self.r_weight * self.loss(input[:,self.abins:,...], target[:,1,...])

class WeightedDiscreteLoss():
    def __init__(self, nbins=19, fmax=47.5, reduction='mean'):
        self.loss = CrossEntropy2d(reduction=reduction)
        assert nbins % 2 == 1, "nbins should be odd"
        self.nbins = nbins
        self.fmax = fmax
        self.step = 2 * fmax / float(nbins)
        self.weight = np.ones((nbins), dtype=np.float32)
        self.weight[int(self.fmax / self.step)] = 0.01
        self.weight = torch.from_numpy(self.weight).cuda()

    def tobin(self, target):
        target = torch.clamp(target, -self.fmax + 1e-3, self.fmax - 1e-3)
        return torch.floor((target + self.fmax) / self.step).type(torch.cuda.LongTensor)

    def __call__(self, input, target):
        target = self.tobin(target)
        assert (target >= 0).all() and (target < self.nbins).all()
        return self.loss(input[:,:self.nbins,...], target[:,0,...]) + self.loss(input[:,self.nbins:,...], target[:,1,...], self.weight)


class CrossEntropy2d(nn.Module):
    def __init__(self, reduction='mean', ignore_label=-1):
        super(CrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.reduction = reduction

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction=self.reduction)
        return loss

#class CrossPixelSimilarityLoss():
#    '''
#        Modified from: https://github.com/lppllppl920/Challenge2018/blob/master/loss.py
#    '''
#    def __init__(self, sigma=0.0036, sampling_size=512):
#        self.sigma = sigma
#        self.sampling_size = sampling_size
#        self.epsilon = 1.0e-15
#        self.embed_norm = True # loss does not decrease no matter it is true or false.
#
#    def __call__(self, embeddings, flows):
#        '''
#            embedding: Variable Nx256xHxW (not hyper-column)
#            flows: Variable Nx2xHxW
#        '''
#        assert flows.size(1) == 2
#
#        # flow normalization
#        positive_mask = (flows > 0)
#        flows = -torch.clamp(torch.log(torch.abs(flows) + 1) / math.log(50. + 1), max=1.)
#        flows[positive_mask] = -flows[positive_mask]
#
#        # embedding normalization
#        if self.embed_norm:
#            embeddings /= torch.norm(embeddings, p=2, dim=1, keepdim=True)
#
#        # Spatially random sampling (512 samples)
#        flows_flatten = flows.view(flows.shape[0], 2, -1)
#        random_locations = Variable(torch.from_numpy(np.array(random.sample(range(flows_flatten.shape[2]), self.sampling_size))).long().cuda())
#        flows_sample = torch.index_select(flows_flatten, 2, random_locations)
#
#        # K_f
#        k_f = self.epsilon + torch.norm(torch.unsqueeze(flows_sample, dim=-1).permute(0, 3, 2, 1) -
#                                        torch.unsqueeze(flows_sample, dim=-1).permute(0, 2, 3, 1), p=2, dim=3,
#                                        keepdim=False) ** 2
#        exp_k_f = torch.exp(-k_f / 2. / self.sigma)
#
#        
#        # mask
#        eye = Variable(torch.unsqueeze(torch.eye(k_f.shape[1]), dim=0).cuda())
#        mask = torch.ones_like(exp_k_f) - eye
#
#        # S_f
#        masked_exp_k_f = torch.mul(mask, exp_k_f) + eye
#        s_f = masked_exp_k_f / torch.sum(masked_exp_k_f, dim=1, keepdim=True)
#
#        # K_theta
#        embeddings_flatten = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1)
#        embeddings_sample = torch.index_select(embeddings_flatten, 2, random_locations)
#        embeddings_sample_norm = torch.norm(embeddings_sample, p=2, dim=1, keepdim=True)
#        k_theta = 0.25 * (torch.matmul(embeddings_sample.permute(0, 2, 1), embeddings_sample)) / (self.epsilon + torch.matmul(embeddings_sample_norm.permute(0, 2, 1), embeddings_sample_norm))
#        exp_k_theta = torch.exp(k_theta)
#
#        # S_theta
#        masked_exp_k_theta = torch.mul(mask, exp_k_theta) + math.exp(-0.75) * eye
#        s_theta = masked_exp_k_theta / torch.sum(masked_exp_k_theta, dim=1, keepdim=True)
#
#        # loss
#        loss = -torch.mean(torch.mul(s_f, torch.log(s_theta)))
#
#        return loss

class CrossPixelSimilarityLoss():
    '''
        Modified from: https://github.com/lppllppl920/Challenge2018/blob/master/loss.py
    '''
    def __init__(self, sigma=0.01, sampling_size=512):
        self.sigma = sigma
        self.sampling_size = sampling_size
        self.epsilon = 1.0e-15
        self.embed_norm = True # loss does not decrease no matter it is true or false.

    def __call__(self, embeddings, flows):
        '''
            embedding: Variable Nx256xHxW (not hyper-column)
            flows: Variable Nx2xHxW
        '''
        assert flows.size(1) == 2

        # flow normalization
        positive_mask = (flows > 0)
        flows = -torch.clamp(torch.log(torch.abs(flows) + 1) / math.log(50. + 1), max=1.)
        flows[positive_mask] = -flows[positive_mask]

        # embedding normalization
        if self.embed_norm:
            embeddings /= torch.norm(embeddings, p=2, dim=1, keepdim=True)

        # Spatially random sampling (512 samples)
        flows_flatten = flows.view(flows.shape[0], 2, -1)
        random_locations = Variable(torch.from_numpy(np.array(random.sample(range(flows_flatten.shape[2]), self.sampling_size))).long().cuda())
        flows_sample = torch.index_select(flows_flatten, 2, random_locations)

        # K_f
        k_f = self.epsilon + torch.norm(torch.unsqueeze(flows_sample, dim=-1).permute(0, 3, 2, 1) -
                                        torch.unsqueeze(flows_sample, dim=-1).permute(0, 2, 3, 1), p=2, dim=3,
                                        keepdim=False) ** 2
        exp_k_f = torch.exp(-k_f / 2. / self.sigma)

        
        # mask
        eye = Variable(torch.unsqueeze(torch.eye(k_f.shape[1]), dim=0).cuda())
        mask = torch.ones_like(exp_k_f) - eye

        # S_f
        masked_exp_k_f = torch.mul(mask, exp_k_f) + eye
        s_f = masked_exp_k_f / torch.sum(masked_exp_k_f, dim=1, keepdim=True)

        # K_theta
        embeddings_flatten = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1)
        embeddings_sample = torch.index_select(embeddings_flatten, 2, random_locations)
        embeddings_sample_norm = torch.norm(embeddings_sample, p=2, dim=1, keepdim=True)
        k_theta = 0.25 * (torch.matmul(embeddings_sample.permute(0, 2, 1), embeddings_sample)) / (self.epsilon + torch.matmul(embeddings_sample_norm.permute(0, 2, 1), embeddings_sample_norm))
        exp_k_theta = torch.exp(k_theta)

        # S_theta
        masked_exp_k_theta = torch.mul(mask, exp_k_theta) + eye
        s_theta = masked_exp_k_theta / torch.sum(masked_exp_k_theta, dim=1, keepdim=True)

        # loss
        loss = -torch.mean(torch.mul(s_f, torch.log(s_theta)))

        return loss


class CrossPixelSimilarityFullLoss():
    '''
        Modified from: https://github.com/lppllppl920/Challenge2018/blob/master/loss.py
    '''
    def __init__(self, sigma=0.01):
        self.sigma = sigma
        self.epsilon = 1.0e-15
        self.embed_norm = True # loss does not decrease no matter it is true or false.

    def __call__(self, embeddings, flows):
        '''
            embedding: Variable Nx256xHxW (not hyper-column)
            flows: Variable Nx2xHxW
        '''
        assert flows.size(1) == 2

        # downsample flow
        factor = flows.shape[2] // embeddings.shape[2]
        flows = nn.functional.avg_pool2d(flows, factor, factor)
        assert flows.shape[2] == embeddings.shape[2]

        # flow normalization
        positive_mask = (flows > 0)
        flows = -torch.clamp(torch.log(torch.abs(flows) + 1) / math.log(50. + 1), max=1.)
        flows[positive_mask] = -flows[positive_mask]

        # embedding normalization
        if self.embed_norm:
            embeddings /= torch.norm(embeddings, p=2, dim=1, keepdim=True)

        # Spatially random sampling (512 samples)
        flows_flatten = flows.view(flows.shape[0], 2, -1)
        #random_locations = Variable(torch.from_numpy(np.array(random.sample(range(flows_flatten.shape[2]), self.sampling_size))).long().cuda())
        #flows_sample = torch.index_select(flows_flatten, 2, random_locations)

        # K_f
        k_f = self.epsilon + torch.norm(torch.unsqueeze(flows_flatten, dim=-1).permute(0, 3, 2, 1) -
                                        torch.unsqueeze(flows_flatten, dim=-1).permute(0, 2, 3, 1), p=2, dim=3,
                                        keepdim=False) ** 2
        exp_k_f = torch.exp(-k_f / 2. / self.sigma)

        
        # mask
        eye = Variable(torch.unsqueeze(torch.eye(k_f.shape[1]), dim=0).cuda())
        mask = torch.ones_like(exp_k_f) - eye

        # S_f
        masked_exp_k_f = torch.mul(mask, exp_k_f) + eye
        s_f = masked_exp_k_f / torch.sum(masked_exp_k_f, dim=1, keepdim=True)

        # K_theta
        embeddings_flatten = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1)
        #embeddings_sample = torch.index_select(embeddings_flatten, 2, random_locations)
        embeddings_flatten_norm = torch.norm(embeddings_flatten, p=2, dim=1, keepdim=True)
        k_theta = 0.25 * (torch.matmul(embeddings_flatten.permute(0, 2, 1), embeddings_flatten)) / (self.epsilon + torch.matmul(embeddings_flatten_norm.permute(0, 2, 1), embeddings_flatten_norm))
        exp_k_theta = torch.exp(k_theta)

        # S_theta
        masked_exp_k_theta = torch.mul(mask, exp_k_theta) + eye
        s_theta = masked_exp_k_theta / torch.sum(masked_exp_k_theta, dim=1, keepdim=True)

        # loss
        loss = -torch.mean(torch.mul(s_f, torch.log(s_theta)))

        return loss


def get_column(embeddings, index, full_size):
    col = []
    for embd in embeddings:
        ind = (index.float() / full_size * embd.size(2)).long()
        col.append(torch.index_select(embd.view(embd.shape[0], embd.shape[1], -1), 2, ind))
    return torch.cat(col, dim=1) # N x coldim x sparsenum

class CrossPixelSimilarityColumnLoss(nn.Module):
    '''
        Modified from: https://github.com/lppllppl920/Challenge2018/blob/master/loss.py
    '''
    def __init__(self, sigma=0.0036, sampling_size=512):
        super(CrossPixelSimilarityColumnLoss, self).__init__()
        self.sigma = sigma
        self.sampling_size = sampling_size
        self.epsilon = 1.0e-15
        self.embed_norm = True # loss does not decrease no matter it is true or false.
        self.mlp = nn.Sequential(
            nn.Linear(96 + 96 + 384 + 256 + 4096, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16))

    def forward(self, feats, flows):
        '''
            embedding: Variable Nx256xHxW (not hyper-column)
            flows: Variable Nx2xHxW
        '''
        assert flows.size(1) == 2

        # flow normalization
        positive_mask = (flows > 0)
        flows = -torch.clamp(torch.log(torch.abs(flows) + 1) / math.log(50. + 1), max=1.)
        flows[positive_mask] = -flows[positive_mask]

        # Spatially random sampling (512 samples)
        flows_flatten = flows.view(flows.shape[0], 2, -1)
        random_locations = Variable(torch.from_numpy(np.array(random.sample(range(flows_flatten.shape[2]), self.sampling_size))).long().cuda())
        flows_sample = torch.index_select(flows_flatten, 2, random_locations)

        # K_f
        k_f = self.epsilon + torch.norm(torch.unsqueeze(flows_sample, dim=-1).permute(0, 3, 2, 1) -
                                        torch.unsqueeze(flows_sample, dim=-1).permute(0, 2, 3, 1), p=2, dim=3,
                                        keepdim=False) ** 2
        exp_k_f = torch.exp(-k_f / 2. / self.sigma)

        
        # mask
        eye = Variable(torch.unsqueeze(torch.eye(k_f.shape[1]), dim=0).cuda())
        mask = torch.ones_like(exp_k_f) - eye

        # S_f
        masked_exp_k_f = torch.mul(mask, exp_k_f) + eye
        s_f = masked_exp_k_f / torch.sum(masked_exp_k_f, dim=1, keepdim=True)


        # column
        column = get_column(feats, random_locations, flows.shape[2])
        embedding = self.mlp(column)
        # K_theta
        embedding_norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
        k_theta = 0.25 * (torch.matmul(embedding.permute(0, 2, 1), embedding)) / (self.epsilon + torch.matmul(embedding_norm.permute(0, 2, 1), embedding_norm))
        exp_k_theta = torch.exp(k_theta)

        # S_theta
        masked_exp_k_theta = torch.mul(mask, exp_k_theta) + math.exp(-0.75) * eye
        s_theta = masked_exp_k_theta / torch.sum(masked_exp_k_theta, dim=1, keepdim=True)

        # loss
        loss = -torch.mean(torch.mul(s_f, torch.log(s_theta)))

        return loss


def print_info(name, var):
    print(name, var.size(), torch.max(var).data.cpu()[0], torch.min(var).data.cpu()[0], torch.mean(var).data.cpu()[0])


def MaskL1Loss(input, target, mask):
    input_size = input.size()
    res = torch.sum(torch.abs(input * mask - target * mask))
    total = torch.sum(mask).item()
    if total > 0:
        res = res / (total * input_size[1])
    return res
