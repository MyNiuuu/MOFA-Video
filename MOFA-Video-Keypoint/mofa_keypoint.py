import argparse
import os
import cv2

import numpy as np
from PIL import Image
import torch
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm import tqdm

from packaging import version

from accelerate.utils import set_seed
from transformers import CLIPVisionModelWithProjection

from diffusers import AutoencoderKLTemporalDecoder
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from pipeline.svdxt_pipeline_ctrlnet_loop import FlowControlNetPipeline
from models.ldmk_ctrlnet import FlowControlNet, CMP_demo

from utils.flow_viz import flow_to_image
from utils.utils import get_sparse_flow, get_cmp_flow, draw_landmarks



# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")



def sample_inputs_face(first_frame, landmarks):

    pc, ph, pw = first_frame.shape
    landmarks = landmarks.unsqueeze(0)

    print(landmarks.shape)

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




def draw_landmarks_cv2(image, landmarks):
    for i, point in enumerate(landmarks):
        cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
    return image


def draw_landmarks_sparseflow(image, landmarks, sparseflow):
    for i, point in enumerate(landmarks):

        R = int(sparseflow[int(point[1]), int(point[0]), 0])
        G = int(sparseflow[int(point[1]), int(point[0]), 1])
        B = int(sparseflow[int(point[1]), int(point[0]), 2])
        cv2.circle(image, (int(point[0]), int(point[1])), 5, (R, G, B), -1)
    return image


def draw_landmarks_first(image, landmarks):
    for i, point in enumerate(landmarks):
        cv2.circle(image, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
    return image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        required=True,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--landmark_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=1
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--save_root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    device = 'cuda'

    set_seed(args.seed)

    print('start loading models...')
    # Load scheduler, tokenizer and models.
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision, variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16")
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )

    controlnet = FlowControlNet.from_pretrained(args.resume_from_checkpoint)
    
    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    print('models loaded.')

    cmp = CMP_demo(
        './models/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/config.yaml',
        42000
    ).to(device)
    cmp.requires_grad_(False)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    controlnet.to(device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
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

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    

    print('configuring pipeline....')
    pipeline = FlowControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        controlnet=controlnet,
        image_encoder=image_encoder,
        vae=vae,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(device)

    first_frame = torch.from_numpy(np.array(Image.open(args.image_path).convert('RGB')) / 255.).permute(2, 0, 1)
    first_frame = first_frame.to(device)
    pc, ph, pw = first_frame.shape
    
    print(first_frame.shape)

    landmarks = np.load(args.landmark_path)  # [200, 68, 2]
    landmarks = landmarks[:args.num_frames]

    flow_len = landmarks.shape[0]

    window_size = args.window_size
    stride = window_size // 2

    # while (flow_len - args.window_size) % stride != 0:
    #     flow_len -= 1

    print('flow length:', flow_len)
    landmarks = landmarks[:flow_len]

    print('start data processing...')

    ldmk_clip = landmarks.copy()

    assert ldmk_clip.ndim == 3

    # 320 because training use 320 * 320 resolution, we want the width of the line in pose_imgs are the same as in the 320 * 320 images
    ldmk_clip[:, :, 0] = ldmk_clip[:, :, 0] / pw * 320
    ldmk_clip[:, :, 1] = ldmk_clip[:, :, 1] / ph * 320
    
    # print(ph, pw)

    pose_imgs = []
    for i in range(ldmk_clip.shape[0]):
        pose_img = draw_landmarks(ldmk_clip[i], 320, 320)
        pose_img = cv2.resize(pose_img, (pw, ph), cv2.INTER_NEAREST)
        pose_imgs.append(pose_img)
    pose_imgs = np.array(pose_imgs)
    pose_imgs = torch.from_numpy(pose_imgs).permute(0, 3, 1, 2).float() / 255.
    pose_imgs = pose_imgs.unsqueeze(0).to(weight_dtype).to(device)

    landmarks = torch.from_numpy(landmarks).to(weight_dtype).to(device)

    val_controlnet_image, val_sparse_optical_flow, \
        val_mask, val_first_frame_384, \
            val_sparse_optical_flow_384, val_mask_384 = sample_inputs_face(first_frame, landmarks)
    
    fb, fl, fc, fh, fw = val_sparse_optical_flow.shape

    val_controlnet_flow = get_cmp_flow(
        cmp, 
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
    pil_val_first_frame = Image.fromarray((first_frame.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8))
    
    print('end data processing...')

    print('start diffusion process...')
    num_frames = landmarks.shape[0]
    val_output = pipeline(
        pil_val_first_frame, 
        pil_val_first_frame, 
        controlnet_flow=val_controlnet_flow,
        landmarks=pose_imgs, 
        window_size=args.window_size, 
        stride=stride, 
        height=ph,
        width=pw,
        num_frames=num_frames,
        decode_chunk_size=8,
        motion_bucket_id=127,
        fps=6,
        noise_aug_strength=0.02,
    )

    video_frames = val_output.frames[0]

    for i in range(num_frames):
        img = video_frames[i]
        video_frames[i] = np.array(img)
    video_frames = np.array(video_frames)

    pose_img_nps = (pose_imgs[0].permute(0, 2, 3, 1).cpu().numpy()*255).astype(np.uint8)
    pose_img_nps_pad = []
    for plen in range(pose_img_nps.shape[0]):
        pose_img_nps_pad.append(pose_img_nps[plen])
    pose_img_nps_pad = np.stack(pose_img_nps_pad)

    cv2_firstframe = cv2.cvtColor(cv2.imread(args.image_path), cv2.COLOR_BGR2RGB)

    viz_landmarks = []
    for k in tqdm(range(len(landmarks))):
        im = draw_landmarks_cv2(video_frames[k].copy(), landmarks[k])
        # im = draw_landmarks_first(im, ff_pose_viz)
        viz_landmarks.append(im)
    # viz_landmarks = [np.uint8(np.ones_like(viz_landmarks[-1]) * 255)] + viz_landmarks
    viz_landmarks = np.stack(viz_landmarks)

    viz_esti_flows = []
    for i in range(val_controlnet_flow.shape[1]):
        temp_flow = val_controlnet_flow[0][i].permute(1, 2, 0)
        viz_esti_flows.append(flow_to_image(temp_flow))
    viz_esti_flows = [np.uint8(np.ones_like(viz_esti_flows[-1]) * 255)] + viz_esti_flows
    viz_esti_flows = np.stack(viz_esti_flows)  # [t-1, h, w, c]
    
    out_nps = []
    for plen in range(video_frames.shape[0]):
        out_nps.append(video_frames[plen])
    out_nps = np.stack(out_nps)

    esti_flow_nps = viz_esti_flows

    first_frames = np.stack([cv2_firstframe] * num_frames)

    total_nps = np.concatenate([
        first_frames, esti_flow_nps, pose_img_nps_pad, viz_landmarks, out_nps
    ], axis=2)
    
    total_path = args.save_root

    os.makedirs(os.path.dirname(total_path), exist_ok=True)
    torchvision.io.write_video(total_path, total_nps, fps=25, video_codec='h264', options={'crf': '10'})

    print(f'saved to {total_path}.')


if __name__ == "__main__":
    main()