import os
import numpy as np
import ffmpeg
import time
import argparse
import random
import torch
from PIL import Image


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def resize_and_crop(image_path, min_size=512, max_size=1024):
    image = Image.open(image_path)
    width, height = image.size

    if width % 64 == 0 and height % 64 == 0:
        return image
    
    min_side = min(width, height)
    max_side = max(width, height)

    # Resize according to the rules
    if min_side < min_size:
        scale_factor = min_size / min_side
    elif min_side > max_size:
        scale_factor = max_size / min_side
    else:
        scale_factor = (min_side // 64) * 64 / min_side

    new_width = round(width * scale_factor)
    new_height = round(height * scale_factor)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    max_side = max(new_width, new_height)

    # Center crop if the longest side is not divisible by 64
    if max_side % 64 != 0:
        crop_width = (new_width // 64) * 64
        crop_height = (new_height // 64) * 64
        print(new_width, new_height)
        print(crop_width, crop_height)
        left = (resized_image.width - crop_width) // 2
        top = (resized_image.height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        cropped_image = resized_image.crop((left, top, right, bottom))
    else:
        cropped_image = resized_image

    print(cropped_image.size)

    return cropped_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--ldmk_render",
        type=str,
        default="aniportrait",
        choices=["sadtalker", "aniportrait"],
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True, 
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True, 
    )
    parser.add_argument(
        "--img_path",
        type=str,
        required=True, 
    )
    parser.add_argument(
        "--save_root",
        type=str,
        required=True, 
    )
    parser.add_argument(
        "--max_frame_len",
        type=int,
        default=125, 
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    set_seed(42)

    args = parse_args()
    
    AUDIO_PATH = args.audio_path
    IMG_PATH = args.img_path
    CKPT_DIR = args.ckpt_dir
    SAVE_ROOT = args.save_root

    WINDOW_SIZE = 25
    MAX_FRAME_LEN = args.max_frame_len
    
    img_name = os.path.splitext(os.path.basename(IMG_PATH))[0]
    audio_name = os.path.splitext(os.path.basename(AUDIO_PATH))[0] + '_' + str(time.time()).split('.')[0]
    ldmk_result_dir = os.path.join(SAVE_ROOT, f"{args.ldmk_render}/ldmk", img_name, audio_name)
    save_video_path = os.path.join(SAVE_ROOT, f'{args.ldmk_render}/out_video', img_name, audio_name + '_noaudio.mp4')

    resize_and_crop(IMG_PATH, min_size=512, max_size=512).save(IMG_PATH)

    if args.ldmk_render == 'sadtalker':
        # assert False
        return_code = os.system(
            f'''
            python sadtalker_audio2pose/inference.py \
                --preprocess full \
                --size 256 \
                --driven_audio {AUDIO_PATH} \
                --source_image {IMG_PATH} \
                --result_dir {ldmk_result_dir} \
                --facerender pirender \
                --verbose \
                --face3dvis
            ''')
        assert return_code == 0
    
    elif args.ldmk_render == 'aniportrait':
        return_code = os.system(
            f'''
            python aniportrait/audio2ldmk.py \
            --ref_image_path {IMG_PATH} \
            --audio_path {AUDIO_PATH} \
            --save_dir {ldmk_result_dir} \
            '''
        )
        assert return_code == 0
    else:
        assert False, "Unsupport landmark generator."

    return_code = os.system(
        f'''
        python mofa_keypoint.py \
            --pretrained_model_name_or_path="ckpts/mofa/stable-video-diffusion-img2vid-xt-1-1" \
            --resume_from_checkpoint={CKPT_DIR} \
            --image_path={IMG_PATH} \
            --landmark_path="{os.path.join(ldmk_result_dir, 'landmarks.npy')}" \
            --num_frames={MAX_FRAME_LEN} \
            --window_size={WINDOW_SIZE} \
            --save_root="{save_video_path}" \
            --mixed_precision="fp16" \
            --seed=41 \
        ''')
    assert return_code == 0
    
    save_av_path = save_video_path.replace('_noaudio', '')
    stream = ffmpeg.input(save_video_path)
    audio = ffmpeg.input(AUDIO_PATH)
    ffmpeg.output(stream.video, audio.audio, save_av_path, vcodec='copy', acodec='aac', shortest=None).run()
    # os.system(f'ffmpeg -i {save_video_path} -i {AUDIO_PATH} -c:v copy -c:a aac -shortest {save_av_path}')
    os.system(f'rm -rf {save_video_path}')
