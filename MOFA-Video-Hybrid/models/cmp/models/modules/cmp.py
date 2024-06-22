import torch
import torch.nn as nn
import models.cmp.models as models


class CMP(nn.Module):

    def __init__(self, params):
        super(CMP, self).__init__()
        img_enc_dim = params['img_enc_dim']
        sparse_enc_dim = params['sparse_enc_dim']
        output_dim = params['output_dim']
        pretrained = params['pretrained_image_encoder']
        decoder_combo = params['decoder_combo']
        self.skip_layer = params['skip_layer']
        if self.skip_layer:
            assert params['flow_decoder'] == "MotionDecoderSkipLayer"

        self.image_encoder = models.backbone.__dict__[params['image_encoder']](
            img_enc_dim, pretrained)
        self.flow_encoder = models.modules.__dict__[params['sparse_encoder']](
            sparse_enc_dim)
        self.flow_decoder = models.modules.__dict__[params['flow_decoder']](
            input_dim=img_enc_dim+sparse_enc_dim,
            output_dim=output_dim, combo=decoder_combo)

    def forward(self, image, sparse):
        sparse_enc = self.flow_encoder(sparse)
        if self.skip_layer:
            img_enc, skip_feat = self.image_encoder(image, ret_feat=True)
            flow_dec = self.flow_decoder(torch.cat((img_enc, sparse_enc), dim=1), skip_feat)
        else:
            img_enc = self.image_encoder(image)
            flow_dec = self.flow_decoder(torch.cat((img_enc, sparse_enc), dim=1))
        return flow_dec


