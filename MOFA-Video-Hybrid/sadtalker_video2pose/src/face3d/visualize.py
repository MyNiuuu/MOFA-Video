# check the sync of 3dmm feature and the audio
import shutil
import cv2
import numpy as np
from src.face3d.models.bfm import ParametricFaceModel
from src.face3d.models.facerecon_model import FaceReconModel
import torch
import subprocess, platform
import scipy.io as scio
from tqdm import tqdm


def draw_landmarks(image, landmarks):
    for i, point in enumerate(landmarks):
        cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
        cv2.putText(image, str(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return image

# draft
def gen_composed_video(args, device, first_frame_coeff, coeff_path, audio_path, save_path, save_lmk_path, crop_info, extended_crop = False):

    coeff_first = scio.loadmat(first_frame_coeff)['full_3dmm']
    info = scio.loadmat(first_frame_coeff)['trans_params'][0]
    print(info)

    coeff_pred = scio.loadmat(coeff_path)['coeff_3dmm']

    # print(coeff_pred.shape)
    # print(coeff_pred[1:, 64:].shape)
    
    if args.still:
        coeff_pred[1:, 64:] = np.stack([coeff_pred[0, 64:]]*coeff_pred[1:, 64:].shape[0])

    # assert False

    coeff_full = np.repeat(coeff_first, coeff_pred.shape[0], axis=0) # 257

    coeff_full[:, 80:144] = coeff_pred[:, 0:64]
    coeff_full[:, 224:227]  = coeff_pred[:, 64:67] # 3 dim translation
    coeff_full[:, 254:]  = coeff_pred[:, 67:] # 3 dim translation

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

    tmp_video_path = '/tmp/face3dtmp.mp4'
    facemodel = FaceReconModel(args)
    im0 = cv2.imread(args.source_image)

    video = cv2.VideoWriter(tmp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (224, 224))

    # since we resize the video, we first need to resize the landmark to the cropped size resolution
    # then, we need to add it back to the original video
    x_scale, y_scale = (ox2 - ox1)/256 , (oy2 - oy1)/256

    W, H = im0.shape[0], im0.shape[1]

    _, _, s, _, _, orig_left, orig_up, orig_crop_size =(info[0], info[1], info[2], info[3], info[4], info[5], info[6], info[7])
    orig_left, orig_up, orig_crop_size = [int(x) for x in (orig_left, orig_up, orig_crop_size)]

    landmark_scale = np.array([[x_scale, y_scale]])
    landmark_shift = np.array([[orig_left, orig_up]])
    landmark_shift2 = np.array([[ox1, oy1]])


    landmarks = []

    for k in tqdm(range(coeff_first.shape[0]), '1st:'):
        cur_coeff_full = torch.tensor(coeff_first, device=device)

        facemodel.forward(cur_coeff_full, device)

        predicted_landmark = facemodel.pred_lm # TODO.
        predicted_landmark = predicted_landmark.cpu().numpy().squeeze()

        predicted_landmark[:, 1] = 224 - predicted_landmark[:, 1]

        predicted_landmark = ((predicted_landmark + landmark_shift) / s[0] * landmark_scale)  + landmark_shift2

        landmarks.append(predicted_landmark)

    print(orig_up, orig_left, orig_crop_size, s)

    for k in tqdm(range(coeff_pred.shape[0]), 'face3d rendering:'):
        cur_coeff_full = torch.tensor(coeff_full[k:k+1], device=device)

        facemodel.forward(cur_coeff_full, device)

        predicted_landmark = facemodel.pred_lm # TODO.
        predicted_landmark = predicted_landmark.cpu().numpy().squeeze()

        predicted_landmark[:, 1] = 224 - predicted_landmark[:, 1]

        predicted_landmark = ((predicted_landmark + landmark_shift) / s[0] * landmark_scale)  + landmark_shift2

        landmarks.append(predicted_landmark)

        rendered_img = facemodel.pred_face
        rendered_img = 255. * rendered_img.cpu().numpy().squeeze().transpose(1,2,0)
        out_img = rendered_img[:, :, :3].astype(np.uint8)

        video.write(np.uint8(out_img[:,:,::-1]))

    video.release()

    # visualize landmarks
    video = cv2.VideoWriter(save_lmk_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (im0.shape[0], im0.shape[1]))

    for k in tqdm(range(len(landmarks)), 'face3d vis:'):
        # im = draw_landmarks(im0.copy(), landmarks[k])
        im = draw_landmarks(np.uint8(np.ones_like(im0)*255), landmarks[k])
        video.write(im)
    video.release()

    shutil.copyfile(args.source_image, save_lmk_path.replace('.mp4', '.png'))

    np.save(save_lmk_path.replace('.mp4', '.npy'), landmarks)

    command = 'ffmpeg -v quiet -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, tmp_video_path, save_path)
    subprocess.call(command, shell=platform.system() != 'Windows')

