import os
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import models.cmp.models as models
import models.cmp.utils as utils


class SingleStageModel(object):

    def __init__(self, params, dist_model=False):
        model_params = params['module']
        self.model = models.modules.__dict__[params['module']['arch']](model_params)
        utils.init_weights(self.model, init_type='xavier')
        self.model.cuda()
        if dist_model:
            self.model = utils.DistModule(self.model)
            self.world_size = dist.get_world_size()
        else:
            self.model = models.modules.FixModule(self.model)
            self.world_size = 1

        if params['optim'] == 'SGD':
            self.optim = torch.optim.SGD(
                self.model.parameters(), lr=params['lr'],
                momentum=0.9, weight_decay=0.0001)
        elif params['optim'] == 'Adam':
            self.optim = torch.optim.Adam(
                self.model.parameters(), lr=params['lr'],
                betas=(params['beta1'], 0.999))
        else:   
            raise Exception("No such optimizer: {}".format(params['optim']))

        cudnn.benchmark = True

    def set_input(self, image_input, sparse_input, flow_target=None, rgb_target=None):
        self.image_input = image_input
        self.sparse_input = sparse_input
        self.flow_target = flow_target
        self.rgb_target = rgb_target

    def eval(self, ret_loss=True):
        pass

    def step(self):
        pass

    def load_state(self, path, Iter, resume=False):
        path = os.path.join(path, "ckpt_iter_{}.pth.tar".format(Iter))

        if resume:
            utils.load_state(path, self.model, self.optim)
        else:
            utils.load_state(path, self.model)

    def load_pretrain(self, load_path):
        utils.load_state(load_path, self.model)

    def save_state(self, path, Iter):
        path = os.path.join(path, "ckpt_iter_{}.pth.tar".format(Iter))

        torch.save({
            'step': Iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict()}, path)

    def switch_to(self, phase):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
