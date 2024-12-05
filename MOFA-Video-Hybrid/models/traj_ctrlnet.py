from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput

from models.controlnet_sdv import ControlNetSDVModel, zero_module
# from unimatch.unimatch.geometry import flow_warp
from models.softsplat import softsplat
# from models.hourglass.dense_motion import DenseMotionNetwork
import models.cmp.models as cmp_models
import models.cmp.utils as cmp_utils

import yaml
import os
import torchvision.transforms as transforms


class ArgObj(object):
    def __init__(self):
        pass


class CMP_demo(nn.Module):
    def __init__(self, configfn, load_iter):
        super().__init__()
        args = ArgObj()
        with open(configfn) as f:
            config = yaml.full_load(f)
        for k, v in config.items():
            setattr(args, k, v)
        setattr(args, 'load_iter', load_iter)
        setattr(args, 'exp_path', os.path.dirname(configfn))
        
        self.model = cmp_models.__dict__[args.model['arch']](args.model, dist_model=False)
        self.model.load_state("{}/checkpoints".format(args.exp_path), args.load_iter, False)        
        self.model.switch_to('eval')
        
        self.data_mean = args.data['data_mean']
        self.data_div = args.data['data_div']
        
        self.img_transform = transforms.Compose([
            transforms.Normalize(self.data_mean, self.data_div)])
        
        self.args = args
        self.fuser = cmp_utils.Fuser(args.model['module']['nbins'], args.model['module']['fmax'])
        torch.cuda.synchronize()

    def run(self, image, sparse, mask):
        dtype = image.dtype
        image = image * 2 - 1
        self.model.set_input(image.float(), torch.cat([sparse, mask], dim=1).float(), None)
        cmp_output = self.model.model(self.model.image_input, self.model.sparse_input)
        flow = self.fuser.convert_flow(cmp_output)
        if flow.shape[2] != self.model.image_input.shape[2]:
            flow = nn.functional.interpolate(
                flow, size=self.model.image_input.shape[2:4],
                mode="bilinear", align_corners=True)

        return flow.to(dtype)  # [b, 2, h, w]

        # tensor_dict = self.model.eval(ret_loss=False)
        # flow = tensor_dict['flow_tensors'][0].cpu().numpy().squeeze().transpose(1,2,0)

        # return flow



class FlowControlNetConditioningEmbeddingSVD(nn.Module):

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):

        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding




class FlowControlNetFirstFrameEncoderLayer(nn.Module):

    def __init__(
        self,
        c_in,
        c_out,
        is_downsample=False
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=2 if is_downsample else 1)
        
    def forward(self, feature):
        '''
            feature: [b, c, h, w]
        '''

        embedding = self.conv_in(feature)
        embedding = F.silu(embedding)

        return embedding



class FlowControlNetFirstFrameEncoder(nn.Module):
    def __init__(
        self,
        c_in=320,
        channels=[320, 640, 1280],
        downsamples=[True, True, True],
        use_zeroconv=True
    ):
        super().__init__()

        self.encoders = nn.ModuleList([])
        self.zeroconvs = nn.ModuleList([])

        for channel, downsample in zip(channels, downsamples):
            self.encoders.append(FlowControlNetFirstFrameEncoderLayer(c_in, channel, is_downsample=downsample))
            self.zeroconvs.append(zero_module(nn.Conv2d(channel, channel, kernel_size=1)) if use_zeroconv else nn.Identity())
            c_in = channel
    
    def forward(self, first_frame):
        feature = first_frame
        deep_features = []
        for encoder, zeroconv in zip(self.encoders, self.zeroconvs):
            feature = encoder(feature)
            # print(feature.shape)
            deep_features.append(zeroconv(feature))
        return deep_features


@dataclass
class FlowControlNetOutput(BaseOutput):
    """
    The output of [`FlowControlNetOutput`].

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
        mid_down_block_re_sample (`torch.Tensor`):
            The activation of the midde block (the lowest sample resolution). Each tensor should be of shape
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`.
            Output can be used to condition the original UNet's middle block activation.
    """

    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor
    controlnet_flow: torch.Tensor
    cmp_output: torch.Tensor


class FlowControlNet(ControlNetSDVModel):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 8,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        up_block_types: Tuple[str] = (
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 768,
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = (5, 10, 10, 20),
        num_frames: int = 25,
        conditioning_channels: int = 3,
        conditioning_embedding_out_channels : Optional[Tuple[int, ...]] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.flow_encoder = FlowControlNetFirstFrameEncoder()

        # time_embed_dim = block_out_channels[0] * 4
        # blocks_time_embed_dim = time_embed_dim
        self.controlnet_cond_embedding = FlowControlNetConditioningEmbeddingSVD(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )
    
    def get_warped_frames(self, first_frame, flows):
        '''
            video_frame: [b, c, h, w]
            flows: [b, t-1, c, h, w]
        '''
        dtype = first_frame.dtype
        warped_frames = []
        for i in range(flows.shape[1]):
            warped_frame = softsplat(tenIn=first_frame.float(), tenFlow=flows[:, i].float(), tenMetric=None, strMode='avg').to(dtype)  # [b, c, h, w]
            warped_frames.append(warped_frame.unsqueeze(1))  # [b, 1, c, h, w]
        warped_frames = torch.cat(warped_frames, dim=1)  # [b, t-1, c, h, w]
        return warped_frames

    def get_cmp_flow(self, frames, sparse_optical_flow, mask):
        '''
            frames: [b, 13, 3, 384, 384] (0, 1) tensor
            sparse_optical_flow: [b, 13, 2, 384, 384] (-384, 384) tensor
            mask: [b, 13, 2, 384, 384] {0, 1} tensor
        '''
        b, t, c, h, w = frames.shape
        assert h == 384 and w == 384
        frames = frames.flatten(0, 1)  # [b*13, 3, 256, 256]
        sparse_optical_flow = sparse_optical_flow.flatten(0, 1)  # [b*13, 2, 256, 256]
        mask = mask.flatten(0, 1)  # [b*13, 2, 256, 256]
        cmp_flow, cmp_output = self.run(frames, sparse_optical_flow, mask)  # [b*13, 2, 256, 256]
        # cmp_flow = self.run(frames.float(), sparse_optical_flow.float(), mask.float())  # [b*13, 2, 256, 256]
        cmp_flow = cmp_flow.reshape(b, t, 2, h, w)
        return cmp_flow, cmp_output
        # return cmp_flow.to(dtype=dtype)
    
    def run(self, image, sparse, mask):
        image = image * 2 - 1
        cmp_output = self.cmp_model(image, torch.cat([sparse, mask], dim=1))
        flow = self.fuser.convert_flow(cmp_output)
        if flow.shape[2] != image.shape[2]:
            flow = nn.functional.interpolate(
                flow, size=image.shape[2:4],
                mode="bilinear", align_corners=True)

        return flow, cmp_output  # [b, 2, h, w]
    
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        controlnet_cond: torch.FloatTensor = None,  # [b, 3, h, w]
        controlnet_flow: torch.FloatTensor = None,  # [b, 13, 2, h, w]
        # controlnet_mask: torch.FloatTensor = None,  # [b, 13, 2, h, w]
        # pixel_values_384: torch.FloatTensor = None,
        # sparse_optical_flow_384: torch.FloatTensor = None,
        # mask_384: torch.FloatTensor = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        guess_mode: bool = False,
        conditioning_scale: float = 1.0,
    ) -> Union[FlowControlNetOutput, Tuple]:
        

        # print(sample.shape)
        # print(controlnet_cond.shape)
        # print(controlnet_flow.shape)

        # assert False
        
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)



        # hourglass_output = self.hourglass_forward(
        #     controlnet_cond, controlnet_sparse_flow, controlnet_mask, controlnet_init_flow)  # [b, l, 3+2+2, h, w]

        # controlnet_flow = controlnet_init_flow + hourglass_output

        # 2. pre-process
        sample = self.conv_in(sample)  # [b*l, 320, h//8, w//8]
        
        # print(controlnet_cond.shape)

        # controlnet cond
        if controlnet_cond != None:
            # embed 成 64*64，和latent一个shape
            controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)  # [b, 320, h//8, w//8]
            # sample = sample + controlnet_cond
        
        # print(controlnet_cond.shape)

        # assert False
        controlnet_cond_features = [controlnet_cond] + self.flow_encoder(controlnet_cond)  # [4]

        # print(controlnet_cond.shape)

        '''
            torch.Size([2, 320, 32, 32])
            torch.Size([2, 320, 16, 16])
            torch.Size([2, 640, 8, 8])
            torch.Size([2, 1280, 4, 4])
        '''

        # for x in controlnet_cond_features:
        #     print(x.shape)
        
        # assert False

        scales = [8, 16, 32, 64]
        scale_flows = {}
        fb, fl, fc, fh, fw = controlnet_flow.shape
        # print(controlnet_flow.shape)
        for scale in scales:
            scaled_flow = F.interpolate(controlnet_flow.reshape(-1, fc, fh, fw), scale_factor=1/scale)
            scaled_flow = scaled_flow.reshape(fb, fl, fc, fh // scale, fw // scale) / scale
            scale_flows[scale] = scaled_flow
        
        # for k in scale_flows.keys():
        #     print(scale_flows[k].shape)
        
        # assert False

        warped_cond_features = []
        for cond_feature in controlnet_cond_features:
            cb, cc, ch, cw = cond_feature.shape
            # print(cond_feature.shape)
            warped_cond_feature = self.get_warped_frames(cond_feature, scale_flows[fh // ch])
            warped_cond_feature = torch.cat([cond_feature.unsqueeze(1), warped_cond_feature], dim=1)  # [b, c, h, w]
            wb, wl, wc, wh, ww = warped_cond_feature.shape
            # print(warped_cond_feature.shape)
            warped_cond_features.append(warped_cond_feature.reshape(wb * wl, wc, wh, ww))
        
        # for x in warped_cond_features:
        #     print(x.shape)
        # assert False
            
        '''
            torch.Size([28, 320, 32, 32])
            torch.Size([28, 320, 16, 16])
            torch.Size([28, 640, 8, 8])
            torch.Size([28, 1280, 4, 4])
        '''

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)


        count = 0
        length = len(warped_cond_features)
        
        # print(sample.shape)
        # print(warped_cond_features[0].shape)

        # add the warped feature in the first scale
        sample = sample + warped_cond_features[count]
        count += 1

        down_block_res_samples = (sample,)

        # print(sample.shape)

        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )
            
            # print(sample.shape)
            # print(warped_cond_features[min(count, length - 1)].shape)
            
            sample = sample + warped_cond_features[min(count, length - 1)]
            count += 1

            down_block_res_samples += res_samples
        
            # print(len(res_samples))
            # for i in range(len(res_samples)):
            #     print(res_samples[i].shape)

            # [28, 320, 32, 32]
            # [28, 320, 32, 32]
            # [28, 320, 16, 16]
                
            # [28, 640, 16, 16]
            # [28, 640, 16, 16]
            # [28, 640, 8, 8]
            
            # [28, 1280, 8, 8]
            # [28, 1280, 8, 8]
            # [28, 1280, 4, 4]
                
            # [28, 1280, 4, 4]
            # [28, 1280, 4, 4]

        # print(sample.shape)
        # print(warped_cond_features[-1].shape)

        # add the warped feature in the last scale
        sample = sample + warped_cond_features[-1]

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )  # [b*l, 1280, h // 64, w // 64]

        # print(sample.shape)

        # assert False

        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling

        down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
        mid_block_res_sample = mid_block_res_sample * conditioning_scale

        # for sample in down_block_res_samples:
        #     print(sample.shape)
        # print(mid_block_res_sample.shape)
        # assert False

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample, controlnet_flow, None)

        return FlowControlNetOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample, controlnet_flow=controlnet_flow, cmp_output=None
        )

