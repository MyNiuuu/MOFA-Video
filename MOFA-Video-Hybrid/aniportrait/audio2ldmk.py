import argparse
import os
# import ffmpeg
import random
import numpy as np
import cv2
import torch
import torchvision
from omegaconf import OmegaConf
from PIL import Image

from src.audio_models.model import Audio2MeshModel
from src.audio_models.pose_model import Audio2PoseModel
from src.utils.audio_util import prepare_audio_feature
from src.utils.mp_utils  import LMKExtractor
from src.utils.pose_util import project_points, smooth_pose_seq


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
        # 选择当前部分的关键点
        indices = np.array(indices) - 1
        current_part_keypoints = keypoints[indices]

        # 绘制关键点
        # for point in current_part_keypoints:
        #     x, y = point
        #     image[y, x, :] = color

        # 绘制连接线
        for i in range(len(indices) - 1):
            x1, y1 = current_part_keypoints[i]
            x2, y2 = current_part_keypoints[i + 1]
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
        
    return image



def convert_ldmk_to_68(mediapipe_ldmk):
    return np.stack([
    # face coutour
    mediapipe_ldmk[:, 234], 
    mediapipe_ldmk[:, 93], 
    mediapipe_ldmk[:, 132], 
    mediapipe_ldmk[:, 58], 
    mediapipe_ldmk[:, 172], 
    mediapipe_ldmk[:, 136], 
    mediapipe_ldmk[:, 150], 
    mediapipe_ldmk[:, 176], 
    mediapipe_ldmk[:, 152], 
    mediapipe_ldmk[:, 400], 
    mediapipe_ldmk[:, 379], 
    mediapipe_ldmk[:, 365], 
    mediapipe_ldmk[:, 397], 
    mediapipe_ldmk[:, 288], 
    mediapipe_ldmk[:, 361], 
    mediapipe_ldmk[:, 323], 
    mediapipe_ldmk[:, 454], 
    # right eyebrow
    mediapipe_ldmk[:, 70], 
    mediapipe_ldmk[:, 63], 
    mediapipe_ldmk[:, 105], 
    mediapipe_ldmk[:, 66], 
    mediapipe_ldmk[:, 107], 
    # left eyebrow
    mediapipe_ldmk[:, 336], 
    mediapipe_ldmk[:, 296], 
    mediapipe_ldmk[:, 334], 
    mediapipe_ldmk[:, 293], 
    mediapipe_ldmk[:, 300], 
    # nose
    mediapipe_ldmk[:, 168], 
    mediapipe_ldmk[:, 6], 
    mediapipe_ldmk[:, 195], 
    mediapipe_ldmk[:, 4], 
    # nose down 
    mediapipe_ldmk[:, 239], 
    mediapipe_ldmk[:, 241], 
    mediapipe_ldmk[:, 19], 
    mediapipe_ldmk[:, 461], 
    mediapipe_ldmk[:, 459], 
    # right eye
    mediapipe_ldmk[:, 33], 
    mediapipe_ldmk[:, 160], 
    mediapipe_ldmk[:, 158], 
    mediapipe_ldmk[:, 133], 
    mediapipe_ldmk[:, 153], 
    mediapipe_ldmk[:, 144], 
    # left eye
    mediapipe_ldmk[:, 362], 
    mediapipe_ldmk[:, 385], 
    mediapipe_ldmk[:, 387], 
    mediapipe_ldmk[:, 263], 
    mediapipe_ldmk[:, 373], 
    mediapipe_ldmk[:, 380],
    # outer lips
    mediapipe_ldmk[:, 61], 
    mediapipe_ldmk[:, 40], 
    mediapipe_ldmk[:, 37], 
    mediapipe_ldmk[:, 0], 
    mediapipe_ldmk[:, 267], 
    mediapipe_ldmk[:, 270], 
    mediapipe_ldmk[:, 291], 
    mediapipe_ldmk[:, 321], 
    mediapipe_ldmk[:, 314], 
    mediapipe_ldmk[:, 17], 
    mediapipe_ldmk[:, 84], 
    mediapipe_ldmk[:, 91], 
    # inner lips
    mediapipe_ldmk[:, 78], 
    mediapipe_ldmk[:, 81], 
    mediapipe_ldmk[:, 13], 
    mediapipe_ldmk[:, 311], 
    mediapipe_ldmk[:, 308], 
    mediapipe_ldmk[:, 402], 
    mediapipe_ldmk[:, 14], 
    mediapipe_ldmk[:, 178],
], axis=1)



# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default='./configs/prompts/animation_audio.yaml')
#     parser.add_argument("-W", type=int, default=512)
#     parser.add_argument("-H", type=int, default=512)
#     parser.add_argument("-L", type=int)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--cfg", type=float, default=3.5)
#     parser.add_argument("--steps", type=int, default=25)
#     parser.add_argument("--fps", type=int, default=30)
#     parser.add_argument("-acc", "--accelerate", action='store_true')
#     parser.add_argument("--fi_step", type=int, default=3)
#     args = parser.parse_args()

#     return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()

    config = OmegaConf.load('aniportrait/configs/config.yaml')

    set_seed(42)

    # if config.weight_dtype == "fp16":
    #     weight_dtype = torch.float16
    # else:
    #     weight_dtype = torch.float32
        
    audio_infer_config = OmegaConf.load(config.audio_inference_config)
    # prepare model
    a2m_model = Audio2MeshModel(audio_infer_config['a2m_model'])
    a2m_model.load_state_dict(torch.load(audio_infer_config['pretrained_model']['a2m_ckpt']), strict=False)
    a2m_model.cuda().eval()

    a2p_model = Audio2PoseModel(audio_infer_config['a2p_model'])
    a2p_model.load_state_dict(torch.load(audio_infer_config['pretrained_model']['a2p_ckpt']), strict=False)
    a2p_model.cuda().eval()

    lmk_extractor = LMKExtractor()

    ref_image_path = args.ref_image_path
    audio_path = args.audio_path
    save_dir = args.save_dir

    ref_image_pil = Image.open(ref_image_path).convert("RGB")
    ref_image_np = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)
    height, width, _ = ref_image_np.shape
    
    face_result = lmk_extractor(ref_image_np)
    assert face_result is not None, "No face detected."
    lmks = face_result['lmks'].astype(np.float32)
    lmks[:, 0] *= width
    lmks[:, 1] *= height

    # print(lmks.shape)

    # assert False
    
    sample = prepare_audio_feature(audio_path, fps=args.fps, wav2vec_model_path=audio_infer_config['a2m_model']['model_path'])
    sample['audio_feature'] = torch.from_numpy(sample['audio_feature']).float().cuda()
    sample['audio_feature'] = sample['audio_feature'].unsqueeze(0)

    # print(sample['audio_feature'].shape)

    # inference
    pred = a2m_model.infer(sample['audio_feature'], sample['seq_len'])
    pred = pred.squeeze().detach().cpu().numpy()
    pred = pred.reshape(pred.shape[0], -1, 3)

    pred = pred + face_result['lmks3d']

    # print(pred.shape)
    
    # assert False

    id_seed = 42
    id_seed = torch.LongTensor([id_seed]).cuda()

    # Currently, only inference up to a maximum length of 10 seconds is supported.
    chunk_duration = 5  # 5 seconds
    chunk_size = args.sr * chunk_duration 


    audio_chunks = list(sample['audio_feature'].split(chunk_size, dim=1))
    seq_len_list = [chunk_duration*args.fps] * (len(audio_chunks) - 1) + [sample['seq_len'] % (chunk_duration*args.fps)]
    audio_chunks[-2] = torch.cat((audio_chunks[-2], audio_chunks[-1]), dim=1)
    seq_len_list[-2] = seq_len_list[-2] + seq_len_list[-1]
    del audio_chunks[-1]
    del seq_len_list[-1]

    # assert False

    pose_seq = []
    for audio, seq_len in zip(audio_chunks, seq_len_list):
        pose_seq_chunk = a2p_model.infer(audio, seq_len, id_seed)
        pose_seq_chunk = pose_seq_chunk.squeeze().detach().cpu().numpy()
        pose_seq_chunk[:, :3] *= 0.5
        pose_seq.append(pose_seq_chunk)
    
    pose_seq = np.concatenate(pose_seq, 0)
    pose_seq = smooth_pose_seq(pose_seq, 7)

    # project 3D mesh to 2D landmark
    projected_vertices = project_points(pred, face_result['trans_mat'], pose_seq, [height, width])
    projected_vertices = np.concatenate([lmks[:468, :2][None, :], projected_vertices], axis=0)
    projected_vertices = convert_ldmk_to_68(projected_vertices)

    # print(projected_vertices.shape)

    pose_images = []
    for i in range(projected_vertices.shape[0]):
        pose_img = draw_landmarks(projected_vertices[i], height, width)
        pose_images.append(pose_img)
    pose_images = np.array(pose_images)

    # print(pose_images.shape)

    ref_image_np = cv2.cvtColor(ref_image_np, cv2.COLOR_BGR2RGB)
    ref_imgs = np.stack([ref_image_np]*(pose_images.shape[0]), axis=0)

    all_np = np.concatenate([ref_imgs, pose_images], axis=2)

    # print(projected_vertices.shape)

    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'landmarks.npy'), projected_vertices)

    torchvision.io.write_video(os.path.join(save_dir, 'landmarks.mp4'), all_np, fps=args.fps, video_codec='h264', options={'crf': '10'})

    # stream = ffmpeg.input(os.path.join(save_dir, 'landmarks.mp4'))
    # audio = ffmpeg.input(args.audio_path)
    # ffmpeg.output(stream.video, audio.audio, os.path.join(save_dir, 'landmarks_audio.mp4'), vcodec='copy', acodec='aac').run()





            

if __name__ == "__main__":
    main()
    