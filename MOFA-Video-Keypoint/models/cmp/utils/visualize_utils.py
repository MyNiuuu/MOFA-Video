import numpy as np

import torch
from . import flowlib

class Fuser(object):
    def __init__(self, nbins, fmax):
        self.nbins = nbins
        self.fmax = fmax
        self.step = 2 * fmax / float(nbins)
        self.mesh = torch.arange(nbins).view(1,-1,1,1).float().cuda() * self.step - fmax + self.step / 2

    def convert_flow(self, flow_prob):
        flow_probx = torch.nn.functional.softmax(flow_prob[:, :self.nbins, :, :], dim=1)
        flow_proby = torch.nn.functional.softmax(flow_prob[:, self.nbins:, :, :], dim=1)
        flow_probx = flow_probx * self.mesh
        flow_proby = flow_proby * self.mesh
        flow = torch.cat([flow_probx.sum(dim=1, keepdim=True), flow_proby.sum(dim=1, keepdim=True)], dim=1)
        return flow

def visualize_tensor_old(image, mask, flow_pred, flow_target, warped, rgb_gen, image_target, image_mean, image_div):
    together = [
        draw_cross(unormalize(image.cpu(), mean=image_mean, div=image_div), mask.cpu(), radius=int(image.size(3) / 50.)),
        flow_to_image(flow_pred.detach().cpu()),
        flow_to_image(flow_target.detach().cpu())]
    if warped is not None:
        together.append(torch.clamp(unormalize(warped.detach().cpu(), mean=image_mean, div=image_div), 0, 255))
    if rgb_gen is not None:
        together.append(torch.clamp(unormalize(rgb_gen.detach().cpu(), mean=image_mean, div=image_div), 0, 255))
    if image_target is not None:
        together.append(torch.clamp(unormalize(image_target.cpu(), mean=image_mean, div=image_div), 0, 255))
    together = torch.cat(together, dim=3)
    return together

def visualize_tensor(image, mask, flow_tensors, common_tensors, rgb_tensors, image_mean, image_div):
    together = [
        draw_cross(unormalize(image.cpu(), mean=image_mean, div=image_div), mask.cpu(), radius=int(image.size(3) / 50.))]
    for ft in flow_tensors:
        together.append(flow_to_image(ft.cpu()))
    for ct in common_tensors:
        together.append(torch.clamp(ct.cpu(), 0, 255))
    for rt in rgb_tensors:
        together.append(torch.clamp(unormalize(rt.cpu(), mean=image_mean, div=image_div), 0, 255))
    together = torch.cat(together, dim=3)
    return together


def unormalize(tensor, mean, div):
    for c, (m, d) in enumerate(zip(mean, div)):
        tensor[:,c,:,:].mul_(d).add_(m)
    return tensor


def flow_to_image(flow):
    flow = flow.numpy()
    flow_img = np.array([flowlib.flow_to_image(fl.transpose((1,2,0))).transpose((2,0,1)) for fl in flow]).astype(np.float32)
    return torch.from_numpy(flow_img)

def shift_tensor(input, offh, offw):
    new = torch.zeros(input.size())
    h = input.size(2)
    w = input.size(3)
    new[:,:,max(0,offh):min(h,h+offh),max(0,offw):min(w,w+offw)] = input[:,:,max(0,-offh):min(h,h-offh),max(0,-offw):min(w,w-offw)]
    return new

def draw_block(mask, radius=5):
    '''
    input:  tensor (NxCxHxW)
    output: block_mask (Nx1xHxW)
    '''
    all_mask = []
    mask = mask[:,0:1,:,:]
    for offh in range(-radius, radius+1):
        for offw in range(-radius, radius+1):
            all_mask.append(shift_tensor(mask, offh, offw))
    block_mask = sum(all_mask)
    block_mask[block_mask > 0] = 1
    return block_mask

def expand_block(sparse, radius=5):
    '''
    input:  sparse (NxCxHxW)
    output: block_sparse (NxCxHxW)
    '''
    all_sparse = []
    for offh in range(-radius, radius+1):
        for offw in range(-radius, radius+1):
            all_sparse.append(shift_tensor(sparse, offh, offw))
    block_sparse = sum(all_sparse)
    return block_sparse

def draw_cross(tensor, mask, radius=5, thickness=2):
    '''
    input:  tensor (NxCxHxW)
            mask (NxXxHxW)
    output: new_tensor (NxCxHxW)
    '''
    all_mask = []
    mask = mask[:,0:1,:,:]
    for off in range(-radius, radius+1):
        for t in range(-thickness, thickness+1):
            all_mask.append(shift_tensor(mask, off, t))
            all_mask.append(shift_tensor(mask, t, off))
    cross_mask = sum(all_mask)
    new_tensor = tensor.clone()
    new_tensor[:,0:1,:,:][cross_mask > 0] = 255.0
    new_tensor[:,1:2,:,:][cross_mask > 0] = 0.0
    new_tensor[:,2:3,:,:][cross_mask > 0] = 0.0
    return new_tensor
