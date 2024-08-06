import os
import uuid
import cv2
from tqdm import tqdm
import yaml
import numpy as np
import warnings
from skimage import img_as_ubyte
import safetensors
import safetensors.torch
warnings.filterwarnings('ignore')


import imageio
import torch
import torchvision

from src.facerender.pirender.config import Config
from src.facerender.pirender.face_model import FaceGenerator

from pydub import AudioSegment
from src.utils.face_enhancer import enhancer_generator_with_len, enhancer_list
from src.utils.paste_pic import paste_pic
from src.utils.videoio import save_video_with_watermark
from src.utils.flow_util import vis_flow
from scipy.io import savemat,loadmat

try:
    import webui  # in webui
    in_webui = True
except:
    in_webui = False

expession = loadmat('expression.mat')
control_dict = {}
for item in ['expression_center', 'expression_mouth', 'expression_eyebrow', 'expression_eyes']:
    control_dict[item] = torch.tensor(expession[item])[0]

class AnimateFromCoeff_PIRender():

    def __init__(self, sadtalker_path, device):

        opt = Config(sadtalker_path['pirender_yaml_path'], None, is_train=False)
        opt.device = device
        self.net_G_ema = FaceGenerator(**opt.gen.param).to(opt.device)
        checkpoint_path = sadtalker_path['pirender_checkpoint']
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.net_G_ema.load_state_dict(checkpoint['net_G_ema'], strict=False)
        print('load [net_G] and [net_G_ema] from {}'.format(checkpoint_path))
        self.net_G = self.net_G_ema.eval()
        self.device = device


    def generate(self, x, video_save_dir, pic_path, crop_info, enhancer=None, background_enhancer=None, preprocess='crop', img_size=256):

        source_image=x['source_image'].type(torch.FloatTensor)
        source_semantics=x['source_semantics'].type(torch.FloatTensor)
        target_semantics=x['target_semantics_list'].type(torch.FloatTensor)

        num = 16

        # import pdb; pdb.set_trace()
        # target_semantics_
        current = target_semantics[0, 0, :64, 0]
        for control_k in range(len(control_dict.keys())):
            listx = list(control_dict.keys())
            control_v = control_dict[listx[control_k]]
            for i in range(num):
                expression = (control_v-current)*i/(num-1)+current
                target_semantics[:, (control_k*num + i):(control_k*num + i+1), :64, :] = expression[None, None, :, None]

        source_image=source_image.to(self.device)
        source_semantics=source_semantics.to(self.device)
        target_semantics=target_semantics.to(self.device)
        frame_num = x['frame_num']

        with torch.no_grad():
            predictions_video = []
            for i in tqdm(range(target_semantics.shape[1]), 'FaceRender:'):
                 predictions_video.append(self.net_G(source_image, target_semantics[:, i])['fake_image'])

        predictions_video = torch.stack(predictions_video, dim=1)
        predictions_video = predictions_video.reshape((-1,)+predictions_video.shape[2:])

        video = []
        for idx in range(len(predictions_video)):
            image = predictions_video[idx]
            image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
            video.append(image)
        result = img_as_ubyte(video)

        ### the generated video is 256x256, so we keep the aspect ratio,
        original_size = crop_info[0]
        if original_size:
            result = [ cv2.resize(result_i,(img_size, int(img_size * original_size[1]/original_size[0]) )) for result_i in result ]

        video_name = x['video_name']  + '.mp4'
        path = os.path.join(video_save_dir, 'temp_'+video_name)

        imageio.mimsave(path, result,  fps=float(25))

        av_path = os.path.join(video_save_dir, video_name)
        return_path = av_path

        audio_path =  x['audio_path']
        audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
        new_audio_path = os.path.join(video_save_dir, audio_name+'.wav')
        start_time = 0
        # cog will not keep the .mp3 filename
        sound = AudioSegment.from_file(audio_path)
        frames = frame_num
        end_time = start_time + frames*1/25*1000
        word1=sound.set_frame_rate(16000)
        word = word1[start_time:end_time]
        word.export(new_audio_path, format="wav")

        save_video_with_watermark(path, new_audio_path, av_path, watermark= False)
        print(f'The generated video is named {video_save_dir}/{video_name}')

        if 'full' in preprocess.lower():
            # only add watermark to the full image.
            video_name_full = x['video_name']  + '_full.mp4'
            full_video_path = os.path.join(video_save_dir, video_name_full)
            return_path = full_video_path
            paste_pic(path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop= True if 'ext' in preprocess.lower() else False)
            print(f'The generated video is named {video_save_dir}/{video_name_full}')
        else:
            full_video_path = av_path

        #### paste back then enhancers
        if enhancer:
            video_name_enhancer = x['video_name']  + '_enhanced.mp4'
            enhanced_path = os.path.join(video_save_dir, 'temp_'+video_name_enhancer)
            av_path_enhancer = os.path.join(video_save_dir, video_name_enhancer)
            return_path = av_path_enhancer

            try:
                enhanced_images_gen_with_len = enhancer_generator_with_len(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))
            except:
                enhanced_images_gen_with_len = enhancer_list(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))

            save_video_with_watermark(enhanced_path, new_audio_path, av_path_enhancer, watermark= False)
            print(f'The generated video is named {video_save_dir}/{video_name_enhancer}')
            os.remove(enhanced_path)

        os.remove(path)
        os.remove(new_audio_path)

        return return_path

    def generate_flow(self, x, video_save_dir, pic_path, crop_info, enhancer=None, background_enhancer=None, preprocess='crop', img_size=256):

        source_image=x['source_image'].type(torch.FloatTensor)
        source_semantics=x['source_semantics'].type(torch.FloatTensor)
        target_semantics=x['target_semantics_list'].type(torch.FloatTensor)


        num = 16

        current = target_semantics[0, 0, :64, 0]
        for control_k in range(len(control_dict.keys())):
            listx = list(control_dict.keys())
            control_v = control_dict[listx[control_k]]
            for i in range(num):
                expression = (control_v-current)*i/(num-1)+current
                target_semantics[:, (control_k*num + i):(control_k*num + i+1), :64, :] = expression[None, None, :, None]

        source_image=source_image.to(self.device)
        source_semantics=source_semantics.to(self.device)
        target_semantics=target_semantics.to(self.device)
        frame_num = x['frame_num']

        with torch.no_grad():
            predictions_video = []
            for i in tqdm(range(target_semantics.shape[1]), 'FaceRender:'):
                 predictions_video.append(self.net_G(source_image, target_semantics[:, i])['flow_field'])

        predictions_video = torch.stack(predictions_video, dim=1)
        predictions_video = predictions_video.reshape((-1,)+predictions_video.shape[2:])

        video = []
        for idx in range(len(predictions_video)):
            image = predictions_video[idx]
            image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
            video.append(image)

        results = np.stack(video, axis=0)

        ### the generated video is 256x256, so we keep the aspect ratio,
        # original_size = crop_info[0]
        # if original_size:
        #     result = [ cv2.resize(result_i,(img_size, int(img_size * original_size[1]/original_size[0]) )) for result_i in result ]
        # results = np.stack(result, axis=0)

        x_name = os.path.basename(pic_path)
        save_name = os.path.join(video_save_dir, x_name + '.flo')
        save_name_flow_vis = os.path.join(video_save_dir, x_name + '.mp4')

        flow_full = paste_flow(results, pic_path, save_name, crop_info, extended_crop= True if 'ext' in preprocess.lower() else False)

        flow_viz = []
        for kk in range(flow_full.shape[0]):
            tmp = vis_flow(flow_full[kk])
            flow_viz.append(tmp)
        flow_viz = np.stack(flow_viz)

        torchvision.io.write_video(save_name_flow_vis, flow_viz, fps=20, video_codec='h264', options={'crf': '10'})

        return save_name_flow_vis


def paste_flow(flows, pic_path, save_name, crop_info, extended_crop=False):

    if not os.path.isfile(pic_path):
        raise ValueError('pic_path must be a valid path to video/image file')
    elif pic_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        # loader for first frame
        full_img = cv2.imread(pic_path)
    else:
        # loader for videos
        video_stream = cv2.VideoCapture(pic_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            break
        full_img = frame
    frame_h = full_img.shape[0]
    frame_w = full_img.shape[1]

    # full images, we only use it as reference for zero init image.

    if len(crop_info) != 3:
        print("you didn't crop the image")
        return
    else:
        r_w, r_h = crop_info[0]
        clx, cly, crx, cry = crop_info[1]
        lx, ly, rx, ry = crop_info[2]
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        # oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
        # oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx

        if extended_crop:
            oy1, oy2, ox1, ox2 = cly, cry, clx, crx
        else:
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx

    # out_tmp = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_w, frame_h))
    # template = np.zeros((frame_h, frame_w, 2)) # full flows
    out_tmp = []
    for crop_frame in tqdm(flows, 'seamlessClone:'):
        p = cv2.resize(crop_frame, (ox2-ox1, oy2 - oy1), interpolation=cv2.INTER_LANCZOS4)

        gen_img = np.zeros((frame_h, frame_w, 2))
        # gen_img = cv2.seamlessClone(p, template, mask, location, cv2.NORMAL_CLONE)
        gen_img[oy1:oy2,ox1:ox2] = p
        out_tmp.append(gen_img)

    np.save(save_name, np.stack(out_tmp))
    return np.stack(out_tmp)