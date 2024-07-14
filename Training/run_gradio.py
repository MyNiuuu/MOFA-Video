import gradio as gr
import numpy as np
import cv2
import os
from PIL import Image, ImageFilter
import uuid
from scipy.interpolate import interp1d, PchipInterpolator
import torchvision
# from utils import *
import time
from tqdm import tqdm
import imageio

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat

from packaging import version

from accelerate.utils import set_seed
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from utils.flow_viz import flow_to_image
from utils.utils import split_filename, image2arr, image2pil, ensure_dirname


output_dir_video = "./outputs/videos"
output_dir_frame = "./outputs/frames"


ensure_dirname(output_dir_video)
ensure_dirname(output_dir_frame)


def divide_points_afterinterpolate(resized_all_points, motion_brush_mask):
    k = resized_all_points.shape[0]
    starts = resized_all_points[:, 0]  # [K, 2]

    in_masks = []
    out_masks = []

    for i in range(k):
        x, y = int(starts[i][1]), int(starts[i][0])
        if motion_brush_mask[x][y] == 255:
            in_masks.append(resized_all_points[i])
        else:
            out_masks.append(resized_all_points[i])
    
    in_masks = np.array(in_masks)
    out_masks = np.array(out_masks)

    return in_masks, out_masks
    

def get_sparseflow_and_mask_forward(
        resized_all_points, 
        n_steps, H, W, 
        is_backward_flow=False
    ):

    K = resized_all_points.shape[0]

    starts = resized_all_points[:, 0]  # [K, 2]

    interpolated_ends = resized_all_points[:, 1:]

    s_flow = np.zeros((K, n_steps, H, W, 2))
    mask = np.zeros((K, n_steps, H, W))

    for k in range(K):
        for i in range(n_steps):
            start, end = starts[k], interpolated_ends[k][i]
            flow = np.int64(end - start) * (-1 if is_backward_flow is True else 1)
            s_flow[k][i][int(start[1]), int(start[0])] = flow
            mask[k][i][int(start[1]), int(start[0])] = 1

    s_flow = np.sum(s_flow, axis=0)
    mask = np.sum(mask, axis=0)

    return s_flow, mask



def init_models(pretrained_model_name_or_path, resume_from_checkpoint, weight_dtype, device='cuda', enable_xformers_memory_efficient_attention=False, allow_tf32=False):

    from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
    from pipeline.pipeline import FlowControlNetPipeline
    from models.svdxt_featureflow_forward_controlnet_s2d_fixcmp_norefine import FlowControlNet, CMP_demo

    print('start loading models...')
    # Load scheduler, tokenizer and models.
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name_or_path, subfolder="image_encoder", revision=None, variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, variant="fp16")
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )

    controlnet = FlowControlNet.from_pretrained(resume_from_checkpoint)

    cmp = CMP_demo(
        './models/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/config.yaml',
        42000
    ).to(device)
    cmp.requires_grad_(False)
    
    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    controlnet.to(device, dtype=weight_dtype)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    pipeline = FlowControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path,
        unet=unet,
        controlnet=controlnet,
        image_encoder=image_encoder,
        vae=vae,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(device)

    print('models loaded.')

    return pipeline, cmp


def interpolate_trajectory(points, n_points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    t = np.linspace(0, 1, len(points))

    fx = PchipInterpolator(t, x)
    fy = PchipInterpolator(t, y)

    new_t = np.linspace(0, 1, n_points)

    new_x = fx(new_t)
    new_y = fy(new_t)
    new_points = list(zip(new_x, new_y))

    return new_points


def visualize_drag_v2(background_image_path, splited_tracks, width, height):
    trajectory_maps = []
    
    background_image = Image.open(background_image_path).convert('RGBA')
    background_image = background_image.resize((width, height))
    w, h = background_image.size
    transparent_background = np.array(background_image)
    transparent_background[:, :, -1] = 128
    transparent_background = Image.fromarray(transparent_background)

    # Create a transparent layer with the same size as the background image
    transparent_layer = np.zeros((h, w, 4))
    for splited_track in splited_tracks:
        if len(splited_track) > 1:
            splited_track = interpolate_trajectory(splited_track, 16)
            splited_track = splited_track[:16]
            for i in range(len(splited_track)-1):
                start_point = (int(splited_track[i][0]), int(splited_track[i][1]))
                end_point = (int(splited_track[i+1][0]), int(splited_track[i+1][1]))
                vx = end_point[0] - start_point[0]
                vy = end_point[1] - start_point[1]
                arrow_length = np.sqrt(vx**2 + vy**2)
                if i == len(splited_track)-2:
                    cv2.arrowedLine(transparent_layer, start_point, end_point, (255, 0, 0, 192), 2, tipLength=8 / arrow_length)
                else:
                    cv2.line(transparent_layer, start_point, end_point, (255, 0, 0, 192), 2)
        else:
            cv2.circle(transparent_layer, (int(splited_track[0][0]), int(splited_track[0][1])), 2, (255, 0, 0, 192), -1)

    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    trajectory_maps.append(trajectory_map)
    return trajectory_maps, transparent_layer


class Drag:
    def __init__(self, device, height, width, model_length):
        self.device = device

        svd_ckpt = "ckpts/stable-video-diffusion-img2vid-xt-1-1"
        mofa_ckpt = "ckpts/controlnet"

        self.device = 'cuda'
        self.weight_dtype = torch.float16

        self.pipeline, self.cmp = init_models(
            svd_ckpt, 
            mofa_ckpt, 
            weight_dtype=self.weight_dtype, 
            device=self.device
        )

        self.height = height
        self.width = width
        self.model_length = model_length

    def get_cmp_flow(self, frames, sparse_optical_flow, mask, brush_mask=None):

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
        cmp_flow = self.cmp.run(frames, sparse_optical_flow, mask)  # [b*13, 2, 256, 256]

        if brush_mask is not None:
            brush_mask = torch.from_numpy(brush_mask) / 255.
            brush_mask = brush_mask.to(cmp_flow.device, dtype=cmp_flow.dtype)
            brush_mask = brush_mask.unsqueeze(0).unsqueeze(0)
            cmp_flow = cmp_flow * brush_mask

        cmp_flow = cmp_flow.reshape(b, t, 2, h, w)
        return cmp_flow
    

    def get_flow(self, pixel_values_384, sparse_optical_flow_384, mask_384, motion_brush_mask=None):

        fb, fl, fc, _, _ = pixel_values_384.shape

        controlnet_flow = self.get_cmp_flow(
            pixel_values_384[:, 0:1, :, :, :].repeat(1, fl, 1, 1, 1), 
            sparse_optical_flow_384, 
            mask_384, motion_brush_mask
        )

        if self.height != 384 or self.width != 384:
            scales = [self.height / 384, self.width / 384]
            controlnet_flow = F.interpolate(controlnet_flow.flatten(0, 1), (self.height, self.width), mode='nearest').reshape(fb, fl, 2, self.height, self.width)
            controlnet_flow[:, :, 0] *= scales[1]
            controlnet_flow[:, :, 1] *= scales[0]
        
        return controlnet_flow
    

    @torch.no_grad()
    def forward_sample(self, input_drag_384_inmask, input_drag_384_outmask, input_first_frame, input_mask_384_inmask, input_mask_384_outmask, in_mask_flag, out_mask_flag, motion_brush_mask=None, ctrl_scale=1., outputs=dict()):
        '''
            input_drag: [1, 13, 320, 576, 2]
            input_drag_384: [1, 13, 384, 384, 2]
            input_first_frame: [1, 3, 320, 576]
        '''

        seed = 42
        num_frames = self.model_length
        
        set_seed(seed)

        input_first_frame_384 = F.interpolate(input_first_frame, (384, 384))
        input_first_frame_384 = input_first_frame_384.repeat(num_frames - 1, 1, 1, 1).unsqueeze(0)
        input_first_frame_pil = Image.fromarray(np.uint8(input_first_frame[0].cpu().permute(1, 2, 0)*255))
        height, width = input_first_frame.shape[-2:]

        input_drag_384_inmask = input_drag_384_inmask.permute(0, 1, 4, 2, 3)  # [1, 13, 2, 384, 384]
        mask_384_inmask = input_mask_384_inmask.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # [1, 13, 2, 384, 384]
        input_drag_384_outmask = input_drag_384_outmask.permute(0, 1, 4, 2, 3)  # [1, 13, 2, 384, 384]
        mask_384_outmask = input_mask_384_outmask.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # [1, 13, 2, 384, 384]
        
        print('start diffusion process...')

        input_drag_384_inmask = input_drag_384_inmask.to(self.device, dtype=self.weight_dtype)
        mask_384_inmask = mask_384_inmask.to(self.device, dtype=self.weight_dtype)
        input_drag_384_outmask = input_drag_384_outmask.to(self.device, dtype=self.weight_dtype)
        mask_384_outmask = mask_384_outmask.to(self.device, dtype=self.weight_dtype)

        input_first_frame_384 = input_first_frame_384.to(self.device, dtype=self.weight_dtype)

        if in_mask_flag:
            flow_inmask = self.get_flow(
                input_first_frame_384, 
                input_drag_384_inmask, mask_384_inmask, motion_brush_mask
            )
        else:
            fb, fl = mask_384_inmask.shape[:2]
            flow_inmask = torch.zeros(fb, fl, 2, self.height, self.width).to(self.device, dtype=self.weight_dtype)

        if out_mask_flag:
            flow_outmask = self.get_flow(
                input_first_frame_384, 
                input_drag_384_outmask, mask_384_outmask
            )
        else:
            fb, fl = mask_384_outmask.shape[:2]
            flow_outmask = torch.zeros(fb, fl, 2, self.height, self.width).to(self.device, dtype=self.weight_dtype)
        
        inmask_no_zero = (flow_inmask != 0).all(dim=2)
        inmask_no_zero = inmask_no_zero.unsqueeze(2).expand_as(flow_inmask)

        controlnet_flow = torch.where(inmask_no_zero, flow_inmask, flow_outmask)

        val_output = self.pipeline(
            input_first_frame_pil, 
            input_first_frame_pil,
            controlnet_flow, 
            height=height,
            width=width,
            num_frames=num_frames,
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=7,
            noise_aug_strength=0.02,
            controlnet_cond_scale=ctrl_scale, 
        )

        video_frames, estimated_flow = val_output.frames[0], val_output.controlnet_flow

        for i in range(num_frames):
            img = video_frames[i]
            video_frames[i] = np.array(img)
        video_frames = torch.from_numpy(np.array(video_frames)).cuda().permute(0, 3, 1, 2).unsqueeze(0) / 255.

        print(video_frames.shape)

        viz_esti_flows = []
        for i in range(estimated_flow.shape[1]):
            temp_flow = estimated_flow[0][i].permute(1, 2, 0)
            viz_esti_flows.append(flow_to_image(temp_flow))
        viz_esti_flows = [np.uint8(np.ones_like(viz_esti_flows[-1]) * 255)] + viz_esti_flows
        viz_esti_flows = np.stack(viz_esti_flows)  # [t-1, h, w, c]

        total_nps = viz_esti_flows

        outputs['logits_imgs'] = video_frames
        outputs['flows'] = torch.from_numpy(total_nps).cuda().permute(0, 3, 1, 2).unsqueeze(0) / 255.

        return outputs

    @torch.no_grad()
    def get_cmp_flow_from_tracking_points(self, tracking_points, motion_brush_mask, first_frame_path):

        original_width, original_height = self.width, self.height

        input_all_points = tracking_points.constructor_args['value']

        if len(input_all_points) == 0 or len(input_all_points[-1]) == 1:
            return np.uint8(np.ones((original_width, original_height, 3))*255)
        
        resized_all_points = [tuple([tuple([int(e1[0]*self.width/original_width), int(e1[1]*self.height/original_height)]) for e1 in e]) for e in input_all_points]
        resized_all_points_384 = [tuple([tuple([int(e1[0]*384/original_width), int(e1[1]*384/original_height)]) for e1 in e]) for e in input_all_points]

        new_resized_all_points = []
        new_resized_all_points_384 = []
        for tnum in range(len(resized_all_points)):
            new_resized_all_points.append(interpolate_trajectory(input_all_points[tnum], self.model_length))
            new_resized_all_points_384.append(interpolate_trajectory(resized_all_points_384[tnum], self.model_length))

        resized_all_points = np.array(new_resized_all_points)
        resized_all_points_384 = np.array(new_resized_all_points_384)

        motion_brush_mask_384 = cv2.resize(motion_brush_mask, (384, 384), cv2.INTER_NEAREST)

        resized_all_points_384_inmask, resized_all_points_384_outmask = \
            divide_points_afterinterpolate(resized_all_points_384, motion_brush_mask_384)

        in_mask_flag = False
        out_mask_flag = False
        
        if resized_all_points_384_inmask.shape[0] != 0:
            in_mask_flag = True
            input_drag_384_inmask, input_mask_384_inmask = \
                get_sparseflow_and_mask_forward(
                    resized_all_points_384_inmask, 
                    self.model_length - 1, 384, 384
                )
        else:
            input_drag_384_inmask, input_mask_384_inmask = \
                np.zeros((self.model_length - 1, 384, 384, 2)), \
                    np.zeros((self.model_length - 1, 384, 384))
        
        if resized_all_points_384_outmask.shape[0] != 0:
            out_mask_flag = True
            input_drag_384_outmask, input_mask_384_outmask = \
                get_sparseflow_and_mask_forward(
                    resized_all_points_384_outmask, 
                    self.model_length - 1, 384, 384
                )
        else:
            input_drag_384_outmask, input_mask_384_outmask = \
                np.zeros((self.model_length - 1, 384, 384, 2)), \
                    np.zeros((self.model_length - 1, 384, 384))

        input_drag_384_inmask = torch.from_numpy(input_drag_384_inmask).unsqueeze(0).to(self.device)  # [1, 13, h, w, 2]
        input_mask_384_inmask = torch.from_numpy(input_mask_384_inmask).unsqueeze(0).to(self.device)  # [1, 13, h, w]
        input_drag_384_outmask = torch.from_numpy(input_drag_384_outmask).unsqueeze(0).to(self.device)  # [1, 13, h, w, 2]
        input_mask_384_outmask = torch.from_numpy(input_mask_384_outmask).unsqueeze(0).to(self.device)  # [1, 13, h, w]

        first_frames_transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.ToTensor(),
        ])

        input_first_frame = image2arr(first_frame_path)
        input_first_frame = repeat(first_frames_transform(input_first_frame), 'c h w -> b c h w', b=1).to(self.device)

        seed = 42
        num_frames = self.model_length
        
        set_seed(seed)

        input_first_frame_384 = F.interpolate(input_first_frame, (384, 384))
        input_first_frame_384 = input_first_frame_384.repeat(num_frames - 1, 1, 1, 1).unsqueeze(0)

        input_drag_384_inmask = input_drag_384_inmask.permute(0, 1, 4, 2, 3)  # [1, 13, 2, 384, 384]
        mask_384_inmask = input_mask_384_inmask.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # [1, 13, 2, 384, 384]
        input_drag_384_outmask = input_drag_384_outmask.permute(0, 1, 4, 2, 3)  # [1, 13, 2, 384, 384]
        mask_384_outmask = input_mask_384_outmask.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # [1, 13, 2, 384, 384]

        input_drag_384_inmask = input_drag_384_inmask.to(self.device, dtype=self.weight_dtype)
        mask_384_inmask = mask_384_inmask.to(self.device, dtype=self.weight_dtype)
        input_drag_384_outmask = input_drag_384_outmask.to(self.device, dtype=self.weight_dtype)
        mask_384_outmask = mask_384_outmask.to(self.device, dtype=self.weight_dtype)

        input_first_frame_384 = input_first_frame_384.to(self.device, dtype=self.weight_dtype)

        if in_mask_flag:
            flow_inmask = self.get_flow(
                input_first_frame_384, 
                input_drag_384_inmask, mask_384_inmask, motion_brush_mask_384
            )
        else:
            fb, fl = mask_384_inmask.shape[:2]
            flow_inmask = torch.zeros(fb, fl, 2, self.height, self.width).to(self.device, dtype=self.weight_dtype)

        if out_mask_flag:
            flow_outmask = self.get_flow(
                input_first_frame_384, 
                input_drag_384_outmask, mask_384_outmask
            )
        else:
            fb, fl = mask_384_outmask.shape[:2]
            flow_outmask = torch.zeros(fb, fl, 2, self.height, self.width).to(self.device, dtype=self.weight_dtype)
        
        inmask_no_zero = (flow_inmask != 0).all(dim=2)
        inmask_no_zero = inmask_no_zero.unsqueeze(2).expand_as(flow_inmask)

        controlnet_flow = torch.where(inmask_no_zero, flow_inmask, flow_outmask)

        controlnet_flow = controlnet_flow[0, -1].permute(1, 2, 0)
        viz_esti_flows = flow_to_image(controlnet_flow)  # [h, w, c]

        return viz_esti_flows

    def run(self, first_frame_path, tracking_points, inference_batch_size, motion_brush_mask, motion_brush_viz, ctrl_scale):
        
        original_width, original_height = self.width, self.height

        input_all_points = tracking_points.constructor_args['value']
        resized_all_points = [tuple([tuple([int(e1[0]*self.width/original_width), int(e1[1]*self.height/original_height)]) for e1 in e]) for e in input_all_points]
        resized_all_points_384 = [tuple([tuple([int(e1[0]*384/original_width), int(e1[1]*384/original_height)]) for e1 in e]) for e in input_all_points]

        new_resized_all_points = []
        new_resized_all_points_384 = []
        for tnum in range(len(resized_all_points)):
            new_resized_all_points.append(interpolate_trajectory(input_all_points[tnum], self.model_length))
            new_resized_all_points_384.append(interpolate_trajectory(resized_all_points_384[tnum], self.model_length))

        resized_all_points = np.array(new_resized_all_points)
        resized_all_points_384 = np.array(new_resized_all_points_384)

        motion_brush_mask_384 = cv2.resize(motion_brush_mask, (384, 384), cv2.INTER_NEAREST)

        resized_all_points_384_inmask, resized_all_points_384_outmask = \
            divide_points_afterinterpolate(resized_all_points_384, motion_brush_mask_384)

        in_mask_flag = False
        out_mask_flag = False
        
        if resized_all_points_384_inmask.shape[0] != 0:
            in_mask_flag = True
            input_drag_384_inmask, input_mask_384_inmask = \
                get_sparseflow_and_mask_forward(
                    resized_all_points_384_inmask, 
                    self.model_length - 1, 384, 384
                )
        else:
            input_drag_384_inmask, input_mask_384_inmask = \
                np.zeros((self.model_length - 1, 384, 384, 2)), \
                    np.zeros((self.model_length - 1, 384, 384))
        
        if resized_all_points_384_outmask.shape[0] != 0:
            out_mask_flag = True
            input_drag_384_outmask, input_mask_384_outmask = \
                get_sparseflow_and_mask_forward(
                    resized_all_points_384_outmask, 
                    self.model_length - 1, 384, 384
                )
        else:
            input_drag_384_outmask, input_mask_384_outmask = \
                np.zeros((self.model_length - 1, 384, 384, 2)), \
                    np.zeros((self.model_length - 1, 384, 384))

        input_drag_384_inmask = torch.from_numpy(input_drag_384_inmask).unsqueeze(0)  # [1, 13, h, w, 2]
        input_mask_384_inmask = torch.from_numpy(input_mask_384_inmask).unsqueeze(0)  # [1, 13, h, w]
        input_drag_384_outmask = torch.from_numpy(input_drag_384_outmask).unsqueeze(0)  # [1, 13, h, w, 2]
        input_mask_384_outmask = torch.from_numpy(input_mask_384_outmask).unsqueeze(0)  # [1, 13, h, w]

        dir, base, ext = split_filename(first_frame_path)
        id = base.split('_')[0]
        
        image_pil = image2pil(first_frame_path)
        image_pil = image_pil.resize((self.width, self.height), Image.BILINEAR).convert('RGB')
        
        visualized_drag, _ = visualize_drag_v2(first_frame_path, resized_all_points, self.width, self.height)

        motion_brush_viz_pil = Image.fromarray(motion_brush_viz.astype(np.uint8)).convert('RGBA')
        visualized_drag = visualized_drag[0].convert('RGBA')
        visualized_drag_brush = Image.alpha_composite(motion_brush_viz_pil, visualized_drag)
        
        first_frames_transform = transforms.Compose([
                        lambda x: Image.fromarray(x),
                        transforms.ToTensor(),
                    ])
        
        outputs = None
        ouput_video_list = []
        ouput_flow_list = []
        num_inference = 1
        for i in tqdm(range(num_inference)):
            if not outputs:
                first_frames = image2arr(first_frame_path)
                first_frames = repeat(first_frames_transform(first_frames), 'c h w -> b c h w', b=inference_batch_size).to(self.device)
            else:
                first_frames = outputs['logits_imgs'][:, -1]
            

            outputs = self.forward_sample(
                input_drag_384_inmask.to(self.device), 
                input_drag_384_outmask.to(self.device), 
                first_frames.to(self.device),
                input_mask_384_inmask.to(self.device),
                input_mask_384_outmask.to(self.device),
                in_mask_flag,
                out_mask_flag, 
                motion_brush_mask_384,
                ctrl_scale)

            ouput_video_list.append(outputs['logits_imgs'])
            ouput_flow_list.append(outputs['flows'])

        hint_path = os.path.join(output_dir_video, str(id), f'{id}_hint.png')
        visualized_drag_brush.save(hint_path)
        
        for i in range(inference_batch_size):
            output_tensor = [ouput_video_list[0][i]]
            flow_tensor = [ouput_flow_list[0][i]]
            output_tensor = torch.cat(output_tensor, dim=0)
            flow_tensor = torch.cat(flow_tensor, dim=0)
            
            outputs_path = os.path.join(output_dir_video, str(id), f's{ctrl_scale}', f'{id}_output.gif')
            flows_path = os.path.join(output_dir_video, str(id), f's{ctrl_scale}', f'{id}_flow.gif')

            outputs_mp4_path = os.path.join(output_dir_video, str(id), f's{ctrl_scale}', f'{id}_output.mp4')
            flows_mp4_path = os.path.join(output_dir_video, str(id), f's{ctrl_scale}', f'{id}_flow.mp4')

            outputs_frames_path = os.path.join(output_dir_frame, str(id), f's{ctrl_scale}', f'{id}_output')
            flows_frames_path = os.path.join(output_dir_frame, str(id), f's{ctrl_scale}', f'{id}_flow')

            os.makedirs(os.path.join(output_dir_video, str(id), f's{ctrl_scale}'), exist_ok=True)
            os.makedirs(os.path.join(outputs_frames_path), exist_ok=True)
            os.makedirs(os.path.join(flows_frames_path), exist_ok=True)

            print(output_tensor.shape)

            output_RGB = output_tensor.permute(0, 2, 3, 1).mul(255).cpu().numpy()
            flow_RGB = flow_tensor.permute(0, 2, 3, 1).mul(255).cpu().numpy()

            torchvision.io.write_video(
                outputs_mp4_path, 
                output_RGB, 
                fps=20, video_codec='h264', options={'crf': '10'}
            )

            torchvision.io.write_video(
                flows_mp4_path, 
                flow_RGB, 
                fps=20, video_codec='h264', options={'crf': '10'}
            )

            imageio.mimsave(outputs_path, np.uint8(output_RGB), fps=20, loop=0)

            imageio.mimsave(flows_path, np.uint8(flow_RGB), fps=20, loop=0)

            for f in range(output_RGB.shape[0]):
                Image.fromarray(np.uint8(output_RGB[f])).save(os.path.join(outputs_frames_path, f'{str(f).zfill(3)}.png'))
                Image.fromarray(np.uint8(flow_RGB[f])).save(os.path.join(flows_frames_path, f'{str(f).zfill(3)}.png'))

        return hint_path, outputs_path, flows_path, outputs_mp4_path, flows_mp4_path


with gr.Blocks() as demo:
    gr.Markdown("""<h1 align="center">MOFA-Video</h1><br>""")

    gr.Markdown("""Official Gradio Demo for <a href='https://myniuuu.github.io/MOFA_Video'><b>MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model</b></a>.<br>""")

    gr.Markdown(
        """
        During the inference, kindly follow these instructions:
        <br>
        1. Use the "Upload Image" button to upload an image. Avoid dragging the image directly into the window. <br>
        2. Proceed to draw trajectories: <br>
            2.1. Click "Add Trajectory" first, then select points on the "Add Trajectory Here" image. The first click sets the starting point. Click multiple points to create a non-linear trajectory. To add a new trajectory, click "Add Trajectory" again and select points on the image. Avoid clicking the "Add Trajectory" button multiple times without clicking points in the image to add the trajectory, as this can lead to errors. <br>
            2.2. After adding each trajectory, an optical flow image will be displayed automatically. Use it as a reference to adjust the trajectory for desired effects (e.g., area, intensity). <br>
            2.3. To delete the latest trajectory, click "Delete Last Trajectory." <br>
            2.4. Choose the Control Scale in the bar. This determines the control intensity. Setting it to 0 means no control (pure generation result of SVD itself), while setting it to 1 results in the strongest control (which will not lead to good results in most cases because of twisting artifacts). A preset value of 0.6 is recommended for most cases. <br>
            2.5. To use the motion brush for restraining the control area of the trajectory, click to add masks on the "Add Motion Brush Here" image. The motion brush restricts the optical flow area derived from the trajectory whose starting point is within the motion brush. The displayed optical flow image will change correspondingly. Adjust the motion brush radius using the "Motion Brush Radius" bar. <br>
        3. Click the "Run" button to animate the image according to the path. <br>
        """
    )

    target_size = 512
    DragNUWA_net = Drag("cuda:0", target_size, target_size, 25)
    first_frame_path = gr.State()
    tracking_points = gr.State([])
    motion_brush_points = gr.State([])
    motion_brush_mask = gr.State()
    motion_brush_viz = gr.State()
    inference_batch_size = gr.State(1)

    def preprocess_image(image):

        image_pil = image2pil(image.name)
        raw_w, raw_h = image_pil.size

        max_edge = min(raw_w, raw_h)
        resize_ratio = target_size / max_edge

        image_pil = image_pil.resize((round(raw_w * resize_ratio), round(raw_h * resize_ratio)), Image.BILINEAR)

        new_w, new_h = image_pil.size
        crop_w = new_w - (new_w % 64)
        crop_h = new_h - (new_h % 64)

        image_pil = transforms.CenterCrop((crop_h, crop_w))(image_pil.convert('RGB'))

        DragNUWA_net.width = crop_w
        DragNUWA_net.height = crop_h

        id = str(time.time()).split('.')[0]
        os.makedirs(os.path.join(output_dir_video, str(id)), exist_ok=True)
        os.makedirs(os.path.join(output_dir_frame, str(id)), exist_ok=True)

        first_frame_path = os.path.join(output_dir_video, str(id), f"{id}_input.png")
        image_pil.save(first_frame_path)

        return first_frame_path, first_frame_path, first_frame_path, gr.State([]), gr.State([]), np.zeros((crop_h, crop_w)), np.zeros((crop_h, crop_w, 4))

    def add_drag(tracking_points):
        if len(tracking_points.constructor_args['value']) != 0 and tracking_points.constructor_args['value'][-1] == []:
            return tracking_points
        tracking_points.constructor_args['value'].append([])
        return tracking_points

    def add_mask(motion_brush_points):
        motion_brush_points.constructor_args['value'].append([])
        return motion_brush_points
    
    def delete_last_drag(tracking_points, first_frame_path, motion_brush_mask):
        if len(tracking_points.constructor_args['value']) > 0:
            tracking_points.constructor_args['value'].pop()
        transparent_background = Image.open(first_frame_path).convert('RGBA')
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))
        for track in tracking_points.constructor_args['value']:
            if len(track) > 1:
                for i in range(len(track)-1):
                    start_point = track[i]
                    end_point = track[i+1]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2)
                    if i == len(track)-2:
                        cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                    else:
                        cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
            else:
                cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)

        viz_flow = DragNUWA_net.get_cmp_flow_from_tracking_points(tracking_points, motion_brush_mask, first_frame_path)

        return tracking_points, trajectory_map, viz_flow
    
    def add_motion_brushes(motion_brush_points, motion_brush_mask, transparent_layer, first_frame_path, radius, tracking_points, evt: gr.SelectData):
        
        transparent_background = Image.open(first_frame_path).convert('RGBA')
        w, h = transparent_background.size

        motion_points = motion_brush_points.constructor_args['value']
        motion_points.append(evt.index)

        x, y = evt.index

        cv2.circle(motion_brush_mask, (x, y), radius, 255, -1)
        cv2.circle(transparent_layer, (x, y), radius, (0, 0, 255, 255), -1)
        
        transparent_layer_pil = Image.fromarray(transparent_layer.astype(np.uint8))
        motion_map = Image.alpha_composite(transparent_background, transparent_layer_pil)

        viz_flow = DragNUWA_net.get_cmp_flow_from_tracking_points(tracking_points, motion_brush_mask, first_frame_path)

        return motion_brush_mask, transparent_layer, motion_map, viz_flow

    def add_tracking_points(tracking_points, first_frame_path, motion_brush_mask, evt: gr.SelectData):

        print(f"You selected {evt.value} at {evt.index} from {evt.target}")
        
        if len(tracking_points.constructor_args['value']) == 0:
            tracking_points.constructor_args['value'].append([])
            
        tracking_points.constructor_args['value'][-1].append(evt.index)

        # print(tracking_points.constructor_args['value'])

        transparent_background = Image.open(first_frame_path).convert('RGBA')
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))
        for track in tracking_points.constructor_args['value']:
            if len(track) > 1:
                for i in range(len(track)-1):
                    start_point = track[i]
                    end_point = track[i+1]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2)
                    if i == len(track)-2:
                        cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                    else:
                        cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
            else:
                cv2.circle(transparent_layer, tuple(track[0]), 3, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)

        viz_flow = DragNUWA_net.get_cmp_flow_from_tracking_points(tracking_points, motion_brush_mask, first_frame_path)

        return tracking_points, trajectory_map, viz_flow

    with gr.Row():
        with gr.Column(scale=2):
            image_upload_button = gr.UploadButton(label="Upload Image",file_types=["image"])
            add_drag_button = gr.Button(value="Add Trajectory")
            run_button = gr.Button(value="Run")
            delete_last_drag_button = gr.Button(value="Delete Last Trajectory")
            brush_radius = gr.Slider(label='Motion Brush Radius', 
                                             minimum=1, 
                                             maximum=100, 
                                             step=1, 
                                             value=10)
            ctrl_scale = gr.Slider(label='Control Scale', 
                                             minimum=0, 
                                             maximum=1., 
                                             step=0.01, 
                                             value=0.6)

        with gr.Column(scale=5):
            input_image = gr.Image(label="Add Trajectory Here",
                                interactive=True)
        with gr.Column(scale=5):
            input_image_mask = gr.Image(label="Add Motion Brush Here",
                                interactive=True)
             
    with gr.Row():   
        with gr.Column(scale=6):
            viz_flow = gr.Image(label="Visualized Flow")
        with gr.Column(scale=6):
            hint_image = gr.Image(label="Visualized Hint Image")
    with gr.Row():
        with gr.Column(scale=6):
            output_video = gr.Image(label="Output Video")
        with gr.Column(scale=6):
            output_flow = gr.Image(label="Output Flow")
    
    with gr.Row():
        with gr.Column(scale=6):
            output_video_mp4 = gr.Video(label="Output Video mp4")
        with gr.Column(scale=6):
            output_flow_mp4 = gr.Video(label="Output Flow mp4")
    
    image_upload_button.upload(preprocess_image, image_upload_button, [input_image, input_image_mask, first_frame_path, tracking_points, motion_brush_points, motion_brush_mask, motion_brush_viz])

    add_drag_button.click(add_drag, tracking_points, tracking_points)

    delete_last_drag_button.click(delete_last_drag, [tracking_points, first_frame_path, motion_brush_mask], [tracking_points, input_image, viz_flow])

    input_image.select(add_tracking_points, [tracking_points, first_frame_path, motion_brush_mask], [tracking_points, input_image, viz_flow])

    input_image_mask.select(add_motion_brushes, [motion_brush_points, motion_brush_mask, motion_brush_viz, first_frame_path, brush_radius, tracking_points], [motion_brush_mask, motion_brush_viz, input_image_mask, viz_flow])

    run_button.click(DragNUWA_net.run, [first_frame_path, tracking_points, inference_batch_size, motion_brush_mask, motion_brush_viz, ctrl_scale], [hint_image, output_video, output_flow, output_video_mp4, output_flow_mp4])

    demo.launch(server_name="127.0.0.1", debug=True, server_port=9080)
