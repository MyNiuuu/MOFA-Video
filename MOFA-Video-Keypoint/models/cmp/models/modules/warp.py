import torch
import torch.nn as nn

class WarpingLayerBWFlow(nn.Module):

    def __init__(self):
        super(WarpingLayerBWFlow, self).__init__()

    def forward(self, image, flow):
        flow_for_grip = torch.zeros_like(flow)
        flow_for_grip[:,0,:,:] = flow[:,0,:,:] / ((flow.size(3) - 1.0) / 2.0)
        flow_for_grip[:,1,:,:] = flow[:,1,:,:] / ((flow.size(2) - 1.0) / 2.0)

        torchHorizontal = torch.linspace(
            -1.0, 1.0, image.size(3)).view(
            1, 1, 1, image.size(3)).expand(
            image.size(0), 1, image.size(2), image.size(3))
        torchVertical = torch.linspace(
            -1.0, 1.0, image.size(2)).view(
            1, 1, image.size(2), 1).expand(
            image.size(0), 1, image.size(2), image.size(3))
        grid = torch.cat([torchHorizontal, torchVertical], 1).cuda()

        grid = (grid + flow_for_grip).permute(0, 2, 3, 1)
        return torch.nn.functional.grid_sample(image, grid)


class WarpingLayerFWFlow(nn.Module):

    def __init__(self):
        super(WarpingLayerFWFlow, self).__init__()
        self.initialized = False

    def forward(self, image, flow, ret_mask = False):
        n, h, w = image.size(0), image.size(2), image.size(3)

        if not self.initialized or n != self.meshx.shape[0] or h * w != self.meshx.shape[1]:
            self.meshx = torch.arange(w).view(1, 1, w).expand(
                n, h, w).contiguous().view(n, -1).cuda()
            self.meshy = torch.arange(h).view(1, h, 1).expand(
                n, h, w).contiguous().view(n, -1).cuda()
            self.warped_image = torch.zeros((n, 3, h, w), dtype=torch.float32).cuda()
            if ret_mask:
                self.hole_mask = torch.ones((n, 1, h, w), dtype=torch.float32).cuda()
            self.initialized = True
        
        v = (flow[:,0,:,:] ** 2 + flow[:,1,:,:] ** 2).view(n, -1)
        _, sortidx = torch.sort(v, dim=1)

        warped_meshx = self.meshx + flow[:,0,:,:].long().view(n, -1)
        warped_meshy = self.meshy + flow[:,1,:,:].long().view(n, -1)
        
        warped_meshx = torch.clamp(warped_meshx, 0, w - 1)
        warped_meshy = torch.clamp(warped_meshy, 0, h - 1)
        
        self.warped_image.zero_()
        if ret_mask:
            self.hole_mask.fill_(1.)
        for i in range(n):
            for c in range(3):
                ind = sortidx[i]
                self.warped_image[i,c,warped_meshy[i][ind],warped_meshx[i][ind]] = image[i,c,self.meshy[i][ind],self.meshx[i][ind]]
            if ret_mask:
                self.hole_mask[i,0,warped_meshy[i],warped_meshx[i]] = 0.
        if ret_mask:
            return self.warped_image, self.hole_mask
        else:
            return self.warped_image
