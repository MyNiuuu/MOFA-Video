import cv2

import numpy as np
import torch


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


@torch.no_grad()
def get_cmp_flow(cmp, frames, sparse_optical_flow, mask):
    '''
        frames: [b, 13, 3, 384, 384] (0, 1) tensor
        sparse_optical_flow: [b, 13, 2, 384, 384] (-384, 384) tensor
        mask: [b, 13, 2, 384, 384] {0, 1} tensor
    '''
    # print(frames.shape)
    dtype = frames.dtype
    b, t, c, h, w = sparse_optical_flow.shape
    assert h == 384 and w == 384
    frames = frames.flatten(0, 1)  # [b*13, 3, 256, 256]
    sparse_optical_flow = sparse_optical_flow.flatten(0, 1)  # [b*13, 2, 256, 256]
    mask = mask.flatten(0, 1)  # [b*13, 2, 256, 256]

    # print(frames.shape)
    # print(sparse_optical_flow.shape)
    # print(mask.shape)

    # assert False

    cmp_flow = []
    for i in range(b*t):
        tmp_flow = cmp.run(frames[i:i+1].float(), sparse_optical_flow[i:i+1].float(), mask[i:i+1].float())  # [b*13, 2, 256, 256]
        cmp_flow.append(tmp_flow)
    cmp_flow = torch.cat(cmp_flow, dim=0)
    cmp_flow = cmp_flow.reshape(b, t, 2, h, w)

    return cmp_flow.to(dtype=dtype)



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
