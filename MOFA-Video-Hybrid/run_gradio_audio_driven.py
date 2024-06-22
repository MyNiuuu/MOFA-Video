import gradio as gr
import numpy as np
import cv2
import os
from PIL import Image
from scipy.interpolate import PchipInterpolator
import torchvision
import time
from tqdm import tqdm
import imageio

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from einops import repeat

from pydub import AudioSegment

from packaging import version

from accelerate.utils import set_seed
from transformers import CLIPVisionModelWithProjection

from diffusers import AutoencoderKLTemporalDecoder
from diffusers.utils.import_utils import is_xformers_available

from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from pipeline.pipeline import FlowControlNetPipeline
from models.traj_ctrlnet import FlowControlNet as DragControlNet, CMP_demo
from models.ldmk_ctrlnet import FlowControlNet as FaceControlNet

from utils.flow_viz import flow_to_image
from utils.utils import split_filename, image2arr, image2pil, ensure_dirname


output_dir = "Output_audio_driven"


ensure_dirname(output_dir)


def draw_landmarks_cv2(image, landmarks):
    for i, point in enumerate(landmarks):
        cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
    return image


def sample_optical_flow(A, B, h, w):
    b, l, k, _ = A.shape

    sparse_optical_flow = torch.zeros((b, l, h, w, 2), dtype=B.dtype, device=B.device)
    mask = torch.zeros((b, l, h, w), dtype=torch.uint8, device=B.device)

    x_coords = A[..., 0].long()
    y_coords = A[..., 1].long()

    x_coords = torch.clip(x_coords, 0, h - 1)
    y_coords = torch.clip(y_coords, 0, w - 1)

    b_idx = torch.arange(b)[:, None, None].repeat(1, l, k)
    l_idx = torch.arange(l)[None, :, None].repeat(b, 1, k)

    sparse_optical_flow[b_idx, l_idx, x_coords, y_coords] = B

    mask[b_idx, l_idx, x_coords, y_coords] = 1

    mask = mask.unsqueeze(-1).repeat(1, 1, 1, 1, 2)

    return sparse_optical_flow, mask


@torch.no_grad()
def get_sparse_flow(landmarks, h, w, t):

    landmarks = torch.flip(landmarks, dims=[3])

    pose_flow = (landmarks - landmarks[:, 0:1].repeat(1, t, 1, 1))[:, 1:]  # 前向光流
    according_poses = landmarks[:, 0:1].repeat(1, t - 1, 1, 1)
    
    pose_flow = torch.flip(pose_flow, dims=[3])
    
    b, t, K, _ = pose_flow.shape

    sparse_optical_flow, mask = sample_optical_flow(according_poses, pose_flow, h, w)

    return sparse_optical_flow.permute(0, 1, 4, 2, 3), mask.permute(0, 1, 4, 2, 3)



def sample_inputs_face(first_frame, landmarks):

    pc, ph, pw = first_frame.shape
    landmarks = landmarks.unsqueeze(0)

    pl = landmarks.shape[1]

    sparse_optical_flow, mask = get_sparse_flow(landmarks, ph, pw, pl)

    if ph != 384 or pw != 384:

        first_frame_384 = F.interpolate(first_frame.unsqueeze(0), (384, 384))  # [3, 384, 384]

        landmarks_384 = torch.zeros_like(landmarks)
        landmarks_384[:, :, :, 0] = landmarks[:, :, :, 0] / pw * 384
        landmarks_384[:, :, :, 1] = landmarks[:, :, :, 1] / ph * 384

        sparse_optical_flow_384, mask_384 = get_sparse_flow(landmarks_384, 384, 384, pl)
    
    else:
        first_frame_384, landmarks_384 = first_frame, landmarks
        sparse_optical_flow_384, mask_384 = sparse_optical_flow, mask
    
    controlnet_image = first_frame.unsqueeze(0)

    return controlnet_image, sparse_optical_flow, mask, first_frame_384, sparse_optical_flow_384, mask_384



PARTS = [
    ('FACE', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], (10, 200, 10)),
    ('LEFT_EYE', [43, 44, 45, 46, 47, 48, 43], (180, 200, 10)),
    ('LEFT_EYEBROW', [23, 24, 25, 26, 27], (180, 220, 10)),
    ('RIGHT_EYE', [37, 38, 39, 40, 41, 42, 37], (10, 200, 180)),
    ('RIGHT_EYEBROW', [18, 19, 20, 21, 22], (10, 220, 180)),
    ('NOSE_UP', [28, 29, 30, 31], (10, 200, 250)),
    ('NOSE_DOWN', [32, 33, 34, 35, 36], (250, 200, 10)),
    ('LIPS_OUTER_BOTTOM_LEFT', [55, 56, 57, 58], (10, 180, 20)),
    ('LIPS_OUTER_BOTTOM_RIGHT', [49, 60, 59, 58], (20, 10, 180)),
    ('LIPS_INNER_BOTTOM_LEFT', [65, 66, 67], (100, 100, 30)),
    ('LIPS_INNER_BOTTOM_RIGHT', [61, 68, 67], (100, 150, 50)),
    ('LIPS_OUTER_TOP_LEFT', [52, 53, 54, 55], (20, 80, 100)),
    ('LIPS_OUTER_TOP_RIGHT', [52, 51, 50, 49], (80, 100, 20)),
    ('LIPS_INNER_TOP_LEFT', [63, 64, 65], (120, 100, 200)),
    ('LIPS_INNER_TOP_RIGHT', [63, 62, 61], (150, 120, 100)),
]


def draw_landmarks(keypoints, h, w):
        
    image = np.zeros((h, w, 3))

    for name, indices, color in PARTS:
        indices = np.array(indices) - 1
        current_part_keypoints = keypoints[indices]

        for i in range(len(indices) - 1):
            x1, y1 = current_part_keypoints[i]
            x2, y2 = current_part_keypoints[i + 1]
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
        
    return image


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

    starts = resized_all_points[:, 0]

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


def init_models(pretrained_model_name_or_path, weight_dtype, device='cuda', enable_xformers_memory_efficient_attention=False, allow_tf32=False):

    drag_ckpt = "./ckpts/mofa/traj_controlnet"
    face_ckpt = "./ckpts/mofa/ldmk_controlnet"

    print('start loading models...')

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

    drag_controlnet = DragControlNet.from_pretrained(drag_ckpt)
    face_controlnet = FaceControlNet.from_pretrained(face_ckpt)

    cmp = CMP_demo(
        './models/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/config.yaml',
        42000
    ).to(device)
    cmp.requires_grad_(False)

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    drag_controlnet.requires_grad_(False)
    face_controlnet.requires_grad_(False)

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    drag_controlnet.to(device, dtype=weight_dtype)
    face_controlnet.to(device, dtype=weight_dtype)

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
        face_controlnet=face_controlnet,
        drag_controlnet=drag_controlnet,
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

        pretrained_model_name_or_path = "./ckpts/mofa/stable-video-diffusion-img2vid-xt-1-1"

        self.device = 'cuda'
        self.weight_dtype = torch.float16

        self.pipeline, self.cmp = init_models(
            pretrained_model_name_or_path, 
            weight_dtype=self.weight_dtype, 
            device=self.device,
        )

        self.height = height
        self.width = width
        self.model_length = model_length

    def get_cmp_flow(self, frames, sparse_optical_flow, mask, brush_mask=None):

        b, t, c, h, w = frames.shape
        assert h == 384 and w == 384
        frames = frames.flatten(0, 1)  # [b*13, 3, 256, 256]
        sparse_optical_flow = sparse_optical_flow.flatten(0, 1)  # [b*13, 2, 256, 256]
        mask = mask.flatten(0, 1)  # [b*13, 2, 256, 256]

        cmp_flow = []
        for i in range(b*t):
            tmp_flow = self.cmp.run(frames[i:i+1], sparse_optical_flow[i:i+1], mask[i:i+1])  # [1, 2, 256, 256]
            cmp_flow.append(tmp_flow)
        cmp_flow = torch.cat(cmp_flow, dim=0)  # [b*13, 2, 256, 256]

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
    def forward_sample(self, save_root, first_frame_path, audio_path, hint_path, input_drag_384_inmask, input_drag_384_outmask, input_first_frame, input_mask_384_inmask, input_mask_384_outmask, in_mask_flag, out_mask_flag, motion_brush_mask_384=None, ldmk_mask_mask_origin=None, ctrl_scale_traj=1., ctrl_scale_ldmk=1., ldmk_render='sadtalker'):

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

        ldmk_controlnet_flow, ldmk_pose_imgs, landmarks, num_frames = self.get_landmarks(save_root, first_frame_path, audio_path, input_first_frame[0], self.model_length, ldmk_render=ldmk_render)

        ldmk_flow_len = ldmk_controlnet_flow.shape[1]
        drag_flow_len = controlnet_flow.shape[1]
        repeat_num = ldmk_flow_len // drag_flow_len + 1
        drag_controlnet_flow = controlnet_flow.repeat(1, repeat_num, 1, 1, 1)
        drag_controlnet_flow = drag_controlnet_flow[:, :ldmk_flow_len]

        ldmk_mask_mask_origin = ldmk_mask_mask_origin.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]

        val_output = self.pipeline(
            input_first_frame_pil, 
            input_first_frame_pil, 

            ldmk_controlnet_flow, 
            ldmk_pose_imgs, 

            drag_controlnet_flow, 
            ldmk_mask_mask_origin, 

            height=height,
            width=width,
            num_frames=num_frames,
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=7,
            noise_aug_strength=0.02,
            ctrl_scale_traj=ctrl_scale_traj, 
            ctrl_scale_ldmk=ctrl_scale_ldmk, 
        )

        video_frames, estimated_flow = val_output.frames[0], val_output.controlnet_flow

        for i in range(num_frames):
            img = video_frames[i]
            video_frames[i] = np.array(img)
        
        video_frames = np.array(video_frames)

        outputs = self.save_video(ldmk_pose_imgs, first_frame_path, hint_path, landmarks, video_frames, estimated_flow, drag_controlnet_flow)

        return outputs

    def save_video(self, pose_imgs, image_path, hint_path, landmarks, video_frames, estimated_flow, drag_controlnet_flow, outputs=dict()):

        pose_img_nps = (pose_imgs[0].permute(0, 2, 3, 1).cpu().numpy()*255).astype(np.uint8)

        cv2_firstframe = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        cv2_hint = cv2.cvtColor(cv2.imread(hint_path), cv2.COLOR_BGR2RGB)

        viz_landmarks = []
        for k in tqdm(range(len(landmarks))):
            im = draw_landmarks_cv2(video_frames[k].copy(), landmarks[k])
            viz_landmarks.append(im)
        viz_landmarks = np.stack(viz_landmarks)

        viz_esti_flows = []
        for i in range(estimated_flow.shape[1]):
            temp_flow = estimated_flow[0][i].permute(1, 2, 0)
            viz_esti_flows.append(flow_to_image(temp_flow))
        viz_esti_flows = [np.uint8(np.ones_like(viz_esti_flows[-1]) * 255)] + viz_esti_flows
        viz_esti_flows = np.stack(viz_esti_flows)  # [t-1, h, w, c]

        viz_drag_flows = []
        for i in range(drag_controlnet_flow.shape[1]):
            temp_flow = drag_controlnet_flow[0][i].permute(1, 2, 0)
            viz_drag_flows.append(flow_to_image(temp_flow))
        viz_drag_flows = [np.uint8(np.ones_like(viz_drag_flows[-1]) * 255)] + viz_drag_flows
        viz_drag_flows = np.stack(viz_drag_flows)  # [t-1, h, w, c]

        out_nps = []
        for plen in range(video_frames.shape[0]):
            out_nps.append(video_frames[plen])
        out_nps = np.stack(out_nps)

        first_frames = np.stack([cv2_firstframe] * out_nps.shape[0])
        hints = np.stack([cv2_hint] * out_nps.shape[0])

        total_nps = np.concatenate([
            first_frames, hints, viz_drag_flows, viz_esti_flows, pose_img_nps, viz_landmarks, out_nps
        ], axis=2)

        video_frames_tensor = torch.from_numpy(video_frames).cuda().permute(0, 3, 1, 2).unsqueeze(0) / 255.
        
        outputs['logits_imgs'] = video_frames_tensor
        outputs['traj_flows'] = torch.from_numpy(viz_drag_flows).cuda().permute(0, 3, 1, 2).unsqueeze(0) / 255.
        outputs['ldmk_flows'] = torch.from_numpy(viz_esti_flows).cuda().permute(0, 3, 1, 2).unsqueeze(0) / 255.
        outputs['viz_ldmk'] = torch.from_numpy(pose_img_nps).cuda().permute(0, 3, 1, 2).unsqueeze(0) / 255.
        outputs['out_with_ldmk'] = torch.from_numpy(viz_landmarks).cuda().permute(0, 3, 1, 2).unsqueeze(0) / 255.
        outputs['total'] = torch.from_numpy(total_nps).cuda().permute(0, 3, 1, 2).unsqueeze(0) / 255.

        return outputs

    @torch.no_grad()
    def get_cmp_flow_from_tracking_points(self, tracking_points, motion_brush_mask, first_frame_path):

        original_width, original_height = self.width, self.height

        flow_div = self.model_length

        input_all_points = tracking_points.constructor_args['value']

        if len(input_all_points) == 0 or len(input_all_points[-1]) == 1:
            return np.uint8(np.ones((original_width, original_height, 3))*255)
        
        resized_all_points = [tuple([tuple([int(e1[0]*self.width/original_width), int(e1[1]*self.height/original_height)]) for e1 in e]) for e in input_all_points]
        resized_all_points_384 = [tuple([tuple([int(e1[0]*384/original_width), int(e1[1]*384/original_height)]) for e1 in e]) for e in input_all_points]

        new_resized_all_points = []
        new_resized_all_points_384 = []
        for tnum in range(len(resized_all_points)):
            new_resized_all_points.append(interpolate_trajectory(input_all_points[tnum], flow_div))
            new_resized_all_points_384.append(interpolate_trajectory(resized_all_points_384[tnum], flow_div))

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
                    flow_div - 1, 384, 384
                )
        else:
            input_drag_384_inmask, input_mask_384_inmask = \
                np.zeros((flow_div - 1, 384, 384, 2)), \
                    np.zeros((flow_div - 1, 384, 384))
        
        if resized_all_points_384_outmask.shape[0] != 0:
            out_mask_flag = True
            input_drag_384_outmask, input_mask_384_outmask = \
                get_sparseflow_and_mask_forward(
                    resized_all_points_384_outmask, 
                    flow_div - 1, 384, 384
                )
        else:
            input_drag_384_outmask, input_mask_384_outmask = \
                np.zeros((flow_div - 1, 384, 384, 2)), \
                    np.zeros((flow_div - 1, 384, 384))

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
        num_frames = flow_div
        
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

        print(controlnet_flow.shape)

        controlnet_flow = controlnet_flow[0, -1].permute(1, 2, 0)
        viz_esti_flows = flow_to_image(controlnet_flow)  # [h, w, c]

        return viz_esti_flows

    @torch.no_grad()
    def get_cmp_flow_landmarks(self, frames, sparse_optical_flow, mask):

        dtype = frames.dtype
        b, t, c, h, w = sparse_optical_flow.shape
        assert h == 384 and w == 384
        frames = frames.flatten(0, 1)  # [b*13, 3, 256, 256]
        sparse_optical_flow = sparse_optical_flow.flatten(0, 1)  # [b*13, 2, 256, 256]
        mask = mask.flatten(0, 1)  # [b*13, 2, 256, 256]

        cmp_flow = []
        for i in range(b*t):
            tmp_flow = self.cmp.run(frames[i:i+1].float(), sparse_optical_flow[i:i+1].float(), mask[i:i+1].float())  # [b*13, 2, 256, 256]
            cmp_flow.append(tmp_flow)
        cmp_flow = torch.cat(cmp_flow, dim=0)
        cmp_flow = cmp_flow.reshape(b, t, 2, h, w)

        return cmp_flow.to(dtype=dtype)

    def audio2landmark(self, audio_path, img_path, ldmk_result_dir, ldmk_render=0):

        if ldmk_render == 'sadtalker':
            return_code = os.system(
                f'''
                python sadtalker_audio2pose/inference.py \
                    --preprocess full \
                    --size 256 \
                    --driven_audio {audio_path} \
                    --source_image {img_path} \
                    --result_dir {ldmk_result_dir} \
                    --facerender pirender \
                    --verbose \
                    --face3dvis
                ''')
            assert return_code == 0, "Errors in generating landmarks! Please trace back up for detailed error report."
        elif ldmk_render == 'aniportrait':
            return_code = os.system(
                f'''
                python aniportrait/audio2ldmk.py \
                --ref_image_path {img_path} \
                --audio_path {audio_path} \
                --save_dir {ldmk_result_dir} \
                '''
            )
            assert return_code == 0, "Errors in generating landmarks! Please trace back up for detailed error report."
        else:
            assert False
    
        return os.path.join(ldmk_result_dir, 'landmarks.npy')


    def get_landmarks(self, save_root, first_frame_path, audio_path, first_frame, num_frames=25, ldmk_render='sadtalker'):

        ldmk_dir = os.path.join(save_root, 'landmarks')
        ldmknpy_dir = self.audio2landmark(audio_path, first_frame_path, ldmk_dir, ldmk_render)

        landmarks = np.load(ldmknpy_dir)
        landmarks = landmarks[:num_frames]  # [25, 68, 2]
        flow_len = landmarks.shape[0]

        ldmk_clip = landmarks.copy()

        assert ldmk_clip.ndim == 3

        ldmk_clip[:, :, 0] = ldmk_clip[:, :, 0] / self.width * 320
        ldmk_clip[:, :, 1] = ldmk_clip[:, :, 1] / self.height * 320

        pose_imgs = []
        for i in range(ldmk_clip.shape[0]):
            pose_img = draw_landmarks(ldmk_clip[i], 320, 320)
            pose_img = cv2.resize(pose_img, (self.width, self.height), cv2.INTER_NEAREST)
            pose_imgs.append(pose_img)
        pose_imgs = np.array(pose_imgs)
        pose_imgs = torch.from_numpy(pose_imgs).permute(0, 3, 1, 2).float() / 255.
        pose_imgs = pose_imgs.unsqueeze(0).to(self.weight_dtype).to(self.device)

        landmarks = torch.from_numpy(landmarks).to(self.weight_dtype).to(self.device)

        val_controlnet_image, val_sparse_optical_flow, \
        val_mask, val_first_frame_384, \
            val_sparse_optical_flow_384, val_mask_384 = sample_inputs_face(first_frame, landmarks)

        fb, fl, fc, fh, fw = val_sparse_optical_flow.shape

        val_controlnet_flow = self.get_cmp_flow_landmarks(
            val_first_frame_384.unsqueeze(0).repeat(1, fl, 1, 1, 1), 
            val_sparse_optical_flow_384, 
            val_mask_384
        )

        if fh != 384 or fw != 384:
            scales = [fh / 384, fw / 384]
            val_controlnet_flow = F.interpolate(val_controlnet_flow.flatten(0, 1), (fh, fw), mode='nearest').reshape(fb, fl, 2, fh, fw)
            val_controlnet_flow[:, :, 0] *= scales[1]
            val_controlnet_flow[:, :, 1] *= scales[0]

        val_controlnet_image = val_controlnet_image.unsqueeze(0).repeat(1, fl, 1, 1, 1)

        return val_controlnet_flow, pose_imgs, landmarks, flow_len
    

    def run(self, first_frame_path, audio_path, tracking_points, motion_brush_mask, motion_brush_viz, ldmk_mask_mask, ldmk_mask_viz, ctrl_scale_traj, ctrl_scale_ldmk, ldmk_render):
        

        timestamp = str(time.time()).split('.')[0]
        save_name = f"trajscale{ctrl_scale_traj}_ldmkscale{ctrl_scale_ldmk}_{ldmk_render}_ts{timestamp}"
        save_root = os.path.join(os.path.dirname(audio_path), save_name)
        os.makedirs(save_root, exist_ok=True)

        
        original_width, original_height = self.width, self.height

        flow_div = self.model_length

        input_all_points = tracking_points.constructor_args['value']

        # print(input_all_points)

        resized_all_points = [tuple([tuple([int(e1[0]*self.width/original_width), int(e1[1]*self.height/original_height)]) for e1 in e]) for e in input_all_points]
        resized_all_points_384 = [tuple([tuple([int(e1[0]*384/original_width), int(e1[1]*384/original_height)]) for e1 in e]) for e in input_all_points]

        new_resized_all_points = []
        new_resized_all_points_384 = []
        for tnum in range(len(resized_all_points)):
            new_resized_all_points.append(interpolate_trajectory(input_all_points[tnum], flow_div))
            new_resized_all_points_384.append(interpolate_trajectory(resized_all_points_384[tnum], flow_div))

        resized_all_points = np.array(new_resized_all_points)
        resized_all_points_384 = np.array(new_resized_all_points_384)

        motion_brush_mask_384 = cv2.resize(motion_brush_mask, (384, 384), cv2.INTER_NEAREST)
        # ldmk_mask_mask_384 = cv2.resize(ldmk_mask_mask, (384, 384), cv2.INTER_NEAREST)

        # motion_brush_mask = torch.from_numpy(motion_brush_mask) / 255.
        # motion_brush_mask = motion_brush_mask.to(self.device)

        ldmk_mask_mask = torch.from_numpy(ldmk_mask_mask) / 255.
        ldmk_mask_mask = ldmk_mask_mask.to(self.device)

        if resized_all_points_384.shape[0] != 0:
            resized_all_points_384_inmask, resized_all_points_384_outmask = \
                divide_points_afterinterpolate(resized_all_points_384, motion_brush_mask_384)
        else:
            resized_all_points_384_inmask = np.array([])
            resized_all_points_384_outmask = np.array([])

        in_mask_flag = False
        out_mask_flag = False
        
        if resized_all_points_384_inmask.shape[0] != 0:
            in_mask_flag = True
            input_drag_384_inmask, input_mask_384_inmask = \
                get_sparseflow_and_mask_forward(
                    resized_all_points_384_inmask, 
                    flow_div - 1, 384, 384
                )
        else:
            input_drag_384_inmask, input_mask_384_inmask = \
                np.zeros((flow_div - 1, 384, 384, 2)), \
                    np.zeros((flow_div - 1, 384, 384))
        
        if resized_all_points_384_outmask.shape[0] != 0:
            out_mask_flag = True
            input_drag_384_outmask, input_mask_384_outmask = \
                get_sparseflow_and_mask_forward(
                    resized_all_points_384_outmask, 
                    flow_div - 1, 384, 384
                )
        else:
            input_drag_384_outmask, input_mask_384_outmask = \
                np.zeros((flow_div - 1, 384, 384, 2)), \
                    np.zeros((flow_div - 1, 384, 384))

        input_drag_384_inmask = torch.from_numpy(input_drag_384_inmask).unsqueeze(0)  # [1, 13, h, w, 2]
        input_mask_384_inmask = torch.from_numpy(input_mask_384_inmask).unsqueeze(0)  # [1, 13, h, w]
        input_drag_384_outmask = torch.from_numpy(input_drag_384_outmask).unsqueeze(0)  # [1, 13, h, w, 2]
        input_mask_384_outmask = torch.from_numpy(input_mask_384_outmask).unsqueeze(0)  # [1, 13, h, w]

        dir, base, ext = split_filename(first_frame_path)
        id = base.split('_')[0]
        
        image_pil = image2pil(first_frame_path)
        image_pil = image_pil.resize((self.width, self.height), Image.BILINEAR).convert('RGBA')
        
        visualized_drag, _ = visualize_drag_v2(first_frame_path, resized_all_points, self.width, self.height)

        motion_brush_viz_pil = Image.fromarray(motion_brush_viz.astype(np.uint8)).convert('RGBA')
        visualized_drag = visualized_drag[0].convert('RGBA')
        ldmk_mask_viz_pil = Image.fromarray(ldmk_mask_viz.astype(np.uint8)).convert('RGBA')

        drag_input = Image.alpha_composite(image_pil, visualized_drag)
        motionbrush_ldmkmask = Image.alpha_composite(motion_brush_viz_pil, ldmk_mask_viz_pil)

        visualized_drag_brush_ldmk_mask = Image.alpha_composite(drag_input, motionbrush_ldmkmask)
        
        first_frames_transform = transforms.Compose([
                        lambda x: Image.fromarray(x),
                        transforms.ToTensor(),
                    ])

        hint_path = os.path.join(save_root, f'hint.png')
        visualized_drag_brush_ldmk_mask.save(hint_path)

        first_frames = image2arr(first_frame_path)
        first_frames = repeat(first_frames_transform(first_frames), 'c h w -> b c h w', b=1).to(self.device)
        
        outputs = self.forward_sample(
            save_root, 
            first_frame_path, 
            audio_path, 
            hint_path, 
            input_drag_384_inmask.to(self.device), 
            input_drag_384_outmask.to(self.device), 
            first_frames.to(self.device),
            input_mask_384_inmask.to(self.device),
            input_mask_384_outmask.to(self.device),
            in_mask_flag,
            out_mask_flag, 
            motion_brush_mask_384, ldmk_mask_mask, 
            ctrl_scale_traj, ctrl_scale_ldmk, ldmk_render=ldmk_render)

        traj_flow_tensor = outputs['traj_flows'][0]  # [25, 3, h, w]
        ldmk_flow_tensor = outputs['ldmk_flows'][0]  # [25, 3, h, w]
        viz_ldmk_tensor = outputs['viz_ldmk'][0]  # [25, 3, h, w]
        out_with_ldmk_tensor = outputs['out_with_ldmk'][0]  # [25, 3, h, w]
        output_tensor = outputs['logits_imgs'][0]  # [25, 3, h, w]
        total_tensor = outputs['total'][0]  # [25, 3, h, w]
        
        traj_flows_path = os.path.join(save_root, f'traj_flow.gif')
        ldmk_flows_path = os.path.join(save_root, f'ldmk_flow.gif')
        viz_ldmk_path = os.path.join(save_root, f'viz_ldmk.gif')
        out_with_ldmk_path = os.path.join(save_root, f'output_w_ldmk.gif')
        outputs_path = os.path.join(save_root, f'output.gif')
        total_path = os.path.join(save_root, f'total.gif')

        traj_flows_path_mp4 = os.path.join(save_root, f'traj_flow.mp4')
        ldmk_flows_path_mp4 = os.path.join(save_root, f'ldmk_flow.mp4')
        viz_ldmk_path_mp4 = os.path.join(save_root, f'viz_ldmk.mp4')
        out_with_ldmk_path_mp4 = os.path.join(save_root, f'output_w_ldmk.mp4')
        outputs_path_mp4 = os.path.join(save_root, f'output.mp4')
        total_path_mp4 = os.path.join(save_root, f'total.mp4')

        # print(output_tensor.shape)

        traj_flow_np = traj_flow_tensor.permute(0, 2, 3, 1).clamp(0, 1).mul(255).cpu().numpy()
        ldmk_flow_np = ldmk_flow_tensor.permute(0, 2, 3, 1).clamp(0, 1).mul(255).cpu().numpy()
        viz_ldmk_np = viz_ldmk_tensor.permute(0, 2, 3, 1).clamp(0, 1).mul(255).cpu().numpy()
        out_with_ldmk_np = out_with_ldmk_tensor.permute(0, 2, 3, 1).clamp(0, 1).mul(255).cpu().numpy()
        output_np = output_tensor.permute(0, 2, 3, 1).clamp(0, 1).mul(255).cpu().numpy()
        total_np = total_tensor.permute(0, 2, 3, 1).clamp(0, 1).mul(255).cpu().numpy()

        torchvision.io.write_video(
            traj_flows_path_mp4, 
            traj_flow_np, 
            fps=20, video_codec='h264', options={'crf': '10'}
        )
        torchvision.io.write_video(
            ldmk_flows_path_mp4, 
            ldmk_flow_np, 
            fps=20, video_codec='h264', options={'crf': '10'}
        )
        torchvision.io.write_video(
            viz_ldmk_path_mp4, 
            viz_ldmk_np, 
            fps=20, video_codec='h264', options={'crf': '10'}
        )
        torchvision.io.write_video(
            out_with_ldmk_path_mp4, 
            out_with_ldmk_np, 
            fps=20, video_codec='h264', options={'crf': '10'}
        )
        torchvision.io.write_video(
            outputs_path_mp4, 
            output_np, 
            fps=20, video_codec='h264', options={'crf': '10'}
        )

        imageio.mimsave(traj_flows_path, np.uint8(traj_flow_np), fps=20, loop=0)
        imageio.mimsave(ldmk_flows_path, np.uint8(ldmk_flow_np), fps=20, loop=0)
        imageio.mimsave(viz_ldmk_path, np.uint8(viz_ldmk_np), fps=20, loop=0)
        imageio.mimsave(out_with_ldmk_path, np.uint8(out_with_ldmk_np), fps=20, loop=0)
        imageio.mimsave(outputs_path, np.uint8(output_np), fps=20, loop=0)

        torchvision.io.write_video(total_path_mp4, total_np, fps=20, video_codec='h264', options={'crf': '10'})
        imageio.mimsave(total_path, np.uint8(total_np), fps=20, loop=0)

        return hint_path, traj_flows_path, ldmk_flows_path, viz_ldmk_path, outputs_path, traj_flows_path_mp4, ldmk_flows_path_mp4, viz_ldmk_path_mp4, outputs_path_mp4


with gr.Blocks() as demo:
    gr.Markdown("""<h1 align="center">MOFA-Video</h1><br>""")

    gr.Markdown("""<h2 align="center">Official Gradio Demo for <a href='https://myniuuu.github.io/MOFA_Video'><b>MOFA-Video</b></a>: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model.</h2>""")

    gr.Markdown(
        """
        <h3> 1. Use the "Upload Image" button to upload an image. Avoid dragging the image directly into the window. </h3>
        <h3> 2. Proceed to trajectory control: </h3>
            2.1. Click "Add Trajectory" first, then select points on the "Add Trajectory Here" image. The first click sets the starting point. Click multiple points to create a non-linear trajectory. To add a new trajectory, click "Add Trajectory" again and select points on the image. <br>
            2.2. After adding each trajectory, an optical flow image will be displayed automatically in "Temporary Trajectory Flow Visualization". Use it as a reference to adjust the trajectory for desired effects (e.g., area, intensity). <br>
            2.3. To delete the latest trajectory, click "Delete Last Trajectory." <br>
            2.4. To use the motion brush for restraining the control area of the trajectory, click to add masks on the "Add Motion Brush Here" image. The motion brush restricts the optical flow area derived from the trajectory whose starting point is within the motion brush. The displayed optical flow image will change correspondingly. Adjust the motion brush radius using the "Motion Brush Radius" slider. <br>
            2.5. Choose the Control scale for trajectory using the "Control Scale for Trajectory" slider. This determines the control intensity of trajectory. Setting it to 0 means no control (pure generation result of SVD itself), while setting it to 1 results in the strongest control (which will not lead to good results in most cases because of twisting artifacts). A preset value of 0.6 is recommended for most cases. <br>
        <h3> 3. Proceed to landmark control from audio: </h3>
            3.1. Use the "Upload Audio" button to upload an audio (currently support .wav and .mp3 extensions). <br>
            3.2. Click to add masks on the "Add Landmark Mask Here" image. This mask restricts the optical flow area derived from the landmarks, which should usually covers the area of the person's head parts, and, if desired, body parts for more natural body movement instead of being stationary. Adjust the landmark brush radius using the "Landmark Brush Radius" slider. <br>
            3.3. Choose the Control scale for landmarks using the "Control Scale for Landmark" slider. This determines the control intensity of landmarks. Different from trajectory controls, a preset value of 1 is recommended for most cases. <br>
            3.4. Choose the landmark renderer to generate landmark sequences from the input audio. The landmark generation codes are based on either <a href='https://github.com/OpenTalker/SadTalker'><b>SadTalker</b></a> or <a href='https://github.com/Zejun-Yang/AniPortrait'><b>AniPortrait</b></a>. We empirically find that SadTalker provides landmarks that follow the audio more precisely in the lips part, while Aniportrait provides more significant lips movement. Note that while pure landmark-based control of MOFA-Video supports long video generation via the periodic sampling strategy, current version of hybrid control only supports short video generation (25 frames), which means that the first 25 frames of the generated landmark sequences are used to obtain the result.
        <h3> 4. Click the "Run" button to animate the image according to the trajectory and the landmark. </h3>
        """
    )
    
    target_size = 512  # NOTICE: changing to lower resolution may impair the performance of the model.
    DragNUWA_net = Drag("cuda:0", target_size, target_size, 25)
    first_frame_path = gr.State()
    audio_path = gr.State()
    tracking_points = gr.State([])
    motion_brush_points = gr.State([])
    motion_brush_mask = gr.State()
    motion_brush_viz = gr.State()
    ldmk_mask_mask = gr.State()
    ldmk_mask_viz = gr.State()

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
        os.makedirs(os.path.join(output_dir, str(id)), exist_ok=True)

        first_frame_path = os.path.join(output_dir, str(id), f"input.png")
        image_pil.save(first_frame_path)

        return first_frame_path, first_frame_path, first_frame_path, first_frame_path, gr.State([]), gr.State([]), np.zeros((crop_h, crop_w)), np.zeros((crop_h, crop_w, 4)), np.zeros((crop_h, crop_w)), np.zeros((crop_h, crop_w, 4))

    def convert_audio_to_wav(input_audio_file, output_wav_file):

        extension = os.path.splitext(os.path.basename(input_audio_file))[-1]

        if extension.lower() == ".mp3":
            audio = AudioSegment.from_mp3(input_audio_file)
        elif extension.lower() == ".wav":
            audio = AudioSegment.from_wav(input_audio_file)
        elif extension.lower() == ".ogg":
            audio = AudioSegment.from_ogg(input_audio_file)
        elif extension.lower() == ".flac":
            audio = AudioSegment.from_file(input_audio_file, "flac")
        else:
            raise ValueError(f"Not supported extension: {extension}")

        audio.export(output_wav_file, format="wav")

    def save_audio(audio, first_frame_path):

        assert first_frame_path is not None, "First upload image, then audio!"

        img_basedir = os.path.dirname(first_frame_path)

        id = str(time.time()).split('.')[0]

        audio_path = os.path.join(img_basedir, f'audio_{str(id)}', 'audio.wav')
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)

        # os.system(f'cp -r {audio.name} {audio_path}')

        convert_audio_to_wav(audio.name, audio_path)

        return audio_path, audio_path

    def add_drag(tracking_points):
        if len(tracking_points.constructor_args['value']) != 0 and tracking_points.constructor_args['value'][-1] == []:
            return tracking_points
        tracking_points.constructor_args['value'].append([])
        return tracking_points
    
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
        cv2.circle(transparent_layer, (x, y), radius, (128, 0, 128, 127), -1)
        
        transparent_layer_pil = Image.fromarray(transparent_layer.astype(np.uint8))
        motion_map = Image.alpha_composite(transparent_background, transparent_layer_pil)

        viz_flow = DragNUWA_net.get_cmp_flow_from_tracking_points(tracking_points, motion_brush_mask, first_frame_path)

        return motion_brush_mask, transparent_layer, motion_map, viz_flow


    def add_ldmk_mask(motion_brush_points, motion_brush_mask, transparent_layer, first_frame_path, radius, evt: gr.SelectData):
        
        transparent_background = Image.open(first_frame_path).convert('RGBA')
        w, h = transparent_background.size

        motion_points = motion_brush_points.constructor_args['value']
        motion_points.append(evt.index)

        x, y = evt.index

        cv2.circle(motion_brush_mask, (x, y), radius, 255, -1)
        cv2.circle(transparent_layer, (x, y), radius, (0, 0, 255, 127), -1)
        
        transparent_layer_pil = Image.fromarray(transparent_layer.astype(np.uint8))
        motion_map = Image.alpha_composite(transparent_background, transparent_layer_pil)

        return motion_brush_mask, transparent_layer, motion_map



    def add_tracking_points(tracking_points, first_frame_path, motion_brush_mask, evt: gr.SelectData):  # SelectData is a subclass of EventData
        print(f"You selected {evt.value} at {evt.index} from {evt.target}")

        if len(tracking_points.constructor_args['value']) == 0:
            tracking_points.constructor_args['value'].append([])
            
        tracking_points.constructor_args['value'][-1].append(evt.index)

        print(tracking_points.constructor_args['value'])

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
        with gr.Column(scale=3):
            image_upload_button = gr.UploadButton(label="Upload Image",file_types=["image"])
            audio_upload_button = gr.UploadButton(label="Upload Audio", file_types=["audio"])
            input_audio = gr.Audio(label="Audio")
        with gr.Column(scale=3):
            add_drag_button = gr.Button(value="Add Trajectory")
            delete_last_drag_button = gr.Button(value="Delete Last Trajectory")
            run_button = gr.Button(value="Run")
        with gr.Column(scale=3):
            motion_brush_radius = gr.Slider(label='Motion Brush Radius', 
                                             minimum=1, 
                                             maximum=200, 
                                             step=1, 
                                             value=10)
            ldmk_mask_radius = gr.Slider(label='Landmark Brush Radius', 
                                             minimum=1, 
                                             maximum=200, 
                                             step=1, 
                                             value=10)
        with gr.Column(scale=3):
            ctrl_scale_traj = gr.Slider(label='Control Scale for Trajectory', 
                                             minimum=0, 
                                             maximum=1., 
                                             step=0.01, 
                                             value=0.6)
            ctrl_scale_ldmk = gr.Slider(label='Control Scale for Landmark', 
                                             minimum=0, 
                                             maximum=1., 
                                             step=0.01, 
                                             value=1.)
            ldmk_render = gr.Radio(label='Landmark Renderer', 
                                    choices=['sadtalker', 'aniportrait'], 
                                    value='aniportrait')

        with gr.Column(scale=4):
            input_image = gr.Image(label="Add Trajectory Here",
                                interactive=True)
        with gr.Column(scale=4):
            motion_brush_image = gr.Image(label="Add Motion Brush Here",
                                interactive=True)
        with gr.Column(scale=4):
            ldmk_mask_image = gr.Image(label="Add Landmark Mask Here",
                                interactive=True)
             
    with gr.Row():   
        with gr.Column(scale=6):
            viz_flow = gr.Image(label="Temporary Trajectory Flow Visualization")
        with gr.Column(scale=6):
            hint_image = gr.Image(label="Final Hint Image")
    
    with gr.Row():    
        with gr.Column(scale=6):
            traj_flows_gif = gr.Image(label="Trajectory Flow GIF")
        with gr.Column(scale=6):
            ldmk_flows_gif = gr.Image(label="Landmark Flow GIF")
    with gr.Row():    
        with gr.Column(scale=6):
            viz_ldmk_gif = gr.Image(label="Landmark Visualization GIF")
        with gr.Column(scale=6):
            outputs_gif = gr.Image(label="Output GIF")
    
    with gr.Row():
        with gr.Column(scale=6):
            traj_flows_mp4 = gr.Video(label="Trajectory Flow MP4")
        with gr.Column(scale=6):
            ldmk_flows_mp4 = gr.Video(label="Landmark Flow MP4")
    with gr.Row():
        with gr.Column(scale=6):
            viz_ldmk_mp4 = gr.Video(label="Landmark Visualization MP4")
        with gr.Column(scale=6):
            outputs_mp4 = gr.Video(label="Output MP4")
    
    image_upload_button.upload(preprocess_image, image_upload_button, [input_image, motion_brush_image, ldmk_mask_image, first_frame_path, tracking_points, motion_brush_points, motion_brush_mask, motion_brush_viz, ldmk_mask_mask, ldmk_mask_viz])

    audio_upload_button.upload(save_audio, [audio_upload_button, first_frame_path], [input_audio, audio_path])

    add_drag_button.click(add_drag, tracking_points, tracking_points)

    delete_last_drag_button.click(delete_last_drag, [tracking_points, first_frame_path, motion_brush_mask], [tracking_points, input_image, viz_flow])

    input_image.select(add_tracking_points, [tracking_points, first_frame_path, motion_brush_mask], [tracking_points, input_image, viz_flow])

    motion_brush_image.select(add_motion_brushes, [motion_brush_points, motion_brush_mask, motion_brush_viz, first_frame_path, motion_brush_radius, tracking_points], [motion_brush_mask, motion_brush_viz, motion_brush_image, viz_flow])

    ldmk_mask_image.select(add_ldmk_mask, [motion_brush_points, ldmk_mask_mask, ldmk_mask_viz, first_frame_path, ldmk_mask_radius], [ldmk_mask_mask, ldmk_mask_viz, ldmk_mask_image])

    run_button.click(DragNUWA_net.run, [first_frame_path, audio_path, tracking_points, motion_brush_mask, motion_brush_viz, ldmk_mask_mask, ldmk_mask_viz, ctrl_scale_traj, ctrl_scale_ldmk, ldmk_render], [hint_image, traj_flows_gif, ldmk_flows_gif, viz_ldmk_gif, outputs_gif, traj_flows_mp4, ldmk_flows_mp4, viz_ldmk_mp4, outputs_mp4])

    # demo.launch(server_name="0.0.0.0", debug=True, server_port=80)
    demo.launch(server_name="127.0.0.1", debug=True, server_port=9080)
