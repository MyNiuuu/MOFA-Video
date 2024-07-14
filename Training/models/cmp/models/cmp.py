import torch
import torch.nn as nn

import models.cmp.losses as losses
import models.cmp.utils as utils

from . import SingleStageModel

class CMP(SingleStageModel):

    def __init__(self, params, dist_model=False):
        super(CMP, self).__init__(params, dist_model)
        model_params = params['module']

        # define loss
        if model_params['flow_criterion'] == 'L1':
            self.flow_criterion = nn.SmoothL1Loss()
        elif model_params['flow_criterion'] == 'L2':
            self.flow_criterion = nn.MSELoss()
        elif model_params['flow_criterion'] == 'DiscreteLoss':
            self.flow_criterion = losses.DiscreteLoss(
                nbins=model_params['nbins'], fmax=model_params['fmax'])
        else:
            raise Exception("No such flow loss: {}".format(model_params['flow_criterion']))

        self.fuser = utils.Fuser(nbins=model_params['nbins'],
                                 fmax=model_params['fmax'])
        self.model_params = model_params

    def eval(self, ret_loss=True):
        with torch.no_grad():
            cmp_output = self.model(self.image_input, self.sparse_input)
        if self.model_params['flow_criterion'] == "DiscreteLoss":
            self.flow = self.fuser.convert_flow(cmp_output)
        else:
            self.flow = cmp_output
        if self.flow.shape[2] != self.image_input.shape[2]:
            self.flow = nn.functional.interpolate(
                self.flow, size=self.image_input.shape[2:4],
                mode="bilinear", align_corners=True)

        ret_tensors = {
            'flow_tensors': [self.flow, self.flow_target],
            'common_tensors': [],
            'rgb_tensors': []} # except for image_input

        if ret_loss:
            if cmp_output.shape[2] != self.flow_target.shape[2]:
                cmp_output = nn.functional.interpolate(
                    cmp_output, size=self.flow_target.shape[2:4],
                    mode="bilinear", align_corners=True)
            loss_flow = self.flow_criterion(cmp_output, self.flow_target) / self.world_size
            return ret_tensors, {'loss_flow': loss_flow}
        else:   
            return ret_tensors

    def step(self):
        cmp_output = self.model(self.image_input, self.sparse_input)
        loss_flow = self.flow_criterion(cmp_output, self.flow_target) / self.world_size
        self.optim.zero_grad()
        loss_flow.backward()
        utils.average_gradients(self.model)
        self.optim.step()
        return {'loss_flow': loss_flow}
