from PIL import Image, ImageOps
import scipy.ndimage as ndimage
import cv2
import random
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy import signal
cv2.ocl.setUseOpenCL(False)

def get_edge(data, blur=False):
    if blur:
        data = cv2.GaussianBlur(data, (3, 3), 1.)
    sobel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]]).astype(np.float32)
    ch_edges = []
    for k in range(data.shape[2]):
        edgex = signal.convolve2d(data[:,:,k], sobel, boundary='symm', mode='same')
        edgey = signal.convolve2d(data[:,:,k], sobel.T, boundary='symm', mode='same')
        ch_edges.append(np.sqrt(edgex**2 + edgey**2))
    return sum(ch_edges)

def get_max(score, bbox):
    u = max(0, bbox[0])
    d = min(score.shape[0], bbox[1])
    l = max(0, bbox[2])
    r = min(score.shape[1], bbox[3])
    return score[u:d,l:r].max()

def nms(score, ks):
    assert ks % 2 == 1
    ret_score = score.copy()
    maxpool = maximum_filter(score, footprint=np.ones((ks, ks)))
    ret_score[score < maxpool] = 0.
    return ret_score

def image_flow_crop(img1, img2, flow, crop_size, phase):
    assert len(crop_size) == 2
    pad_h = max(crop_size[0] - img1.height, 0)
    pad_w = max(crop_size[1] - img1.width, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        flow_expand = np.zeros((img1.height + pad_h, img1.width + pad_w, 2), dtype=np.float32)
        flow_expand[pad_h_half:pad_h_half+img1.height, pad_w_half:pad_w_half+img1.width, :] = flow
        flow = flow_expand
        border = (pad_w_half, pad_h_half, pad_w - pad_w_half, pad_h - pad_h_half)
        img1 = ImageOps.expand(img1, border=border, fill=(0,0,0))
        img2 = ImageOps.expand(img2, border=border, fill=(0,0,0))
    if phase == 'train':
        hoff = int(np.random.rand() * (img1.height - crop_size[0]))
        woff = int(np.random.rand() * (img1.width - crop_size[1]))
    else:
        hoff = (img1.height - crop_size[0]) // 2
        woff = (img1.width - crop_size[1]) // 2

    img1 = img1.crop((woff, hoff, woff+crop_size[1], hoff+crop_size[0]))
    img2 = img2.crop((woff, hoff, woff+crop_size[1], hoff+crop_size[0]))
    flow = flow[hoff:hoff+crop_size[0], woff:woff+crop_size[1], :]
    offset = (hoff, woff)
    return img1, img2, flow, offset

def image_crop(img, crop_size):
    pad_h = max(crop_size[0] - img.height, 0)
    pad_w = max(crop_size[1] - img.width, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        border = (pad_w_half, pad_h_half, pad_w - pad_w_half, pad_h - pad_h_half)
        img = ImageOps.expand(img, border=border, fill=(0,0,0))
    hoff = (img.height - crop_size[0]) // 2
    woff = (img.width - crop_size[1]) // 2
    return img.crop((woff, hoff, woff+crop_size[1], hoff+crop_size[0])), (pad_w_half, pad_h_half)

def image_flow_resize(img1, img2, flow, short_size=None, long_size=None):
    assert (short_size is None) ^ (long_size is None)
    w, h = img1.width, img1.height
    if short_size is not None:
        if w < h:
            neww = short_size
            newh = int(short_size / float(w) * h)
        else:
            neww = int(short_size / float(h) * w)
            newh = short_size
    else:
        if w < h:
            neww = int(long_size / float(h) * w)
            newh = long_size
        else:
            neww = long_size
            newh = int(long_size / float(w) * h)
    img1 = img1.resize((neww, newh), Image.BICUBIC)
    img2 = img2.resize((neww, newh), Image.BICUBIC)
    ratio = float(newh) / h
    flow = cv2.resize(flow.copy(), (neww, newh), interpolation=cv2.INTER_LINEAR) * ratio
    return img1, img2, flow, ratio

def image_resize(img, short_size=None, long_size=None):
    assert (short_size is None) ^ (long_size is None)
    w, h = img.width, img.height
    if short_size is not None:
        if w < h:
            neww = short_size
            newh = int(short_size / float(w) * h)
        else:
            neww = int(short_size / float(h) * w)
            newh = short_size
    else:
        if w < h:
            neww = int(long_size / float(h) * w)
            newh = long_size
        else:
            neww = long_size
            newh = int(long_size / float(w) * h)
    img = img.resize((neww, newh), Image.BICUBIC)
    return img, [w, h]


def image_pose_crop(img, posemap, crop_size, scale):
    assert len(crop_size) == 2
    assert crop_size[0] <= img.height
    assert crop_size[1] <= img.width
    hoff = (img.height - crop_size[0]) // 2
    woff = (img.width - crop_size[1]) // 2
    img = img.crop((woff, hoff, woff+crop_size[1], hoff+crop_size[0]))
    posemap = posemap[hoff//scale:hoff//scale+crop_size[0]//scale, woff//scale:woff//scale+crop_size[1]//scale,:]
    return img, posemap

def neighbor_elim(ph, pw, d):
    valid = np.ones((len(ph))).astype(np.int)
    h_dist = np.fabs(np.tile(ph[:,np.newaxis], [1,len(ph)]) - np.tile(ph.T[np.newaxis,:], [len(ph),1]))
    w_dist = np.fabs(np.tile(pw[:,np.newaxis], [1,len(pw)]) - np.tile(pw.T[np.newaxis,:], [len(pw),1]))
    idx1, idx2 = np.where((h_dist < d) & (w_dist < d))
    for i,j in zip(idx1, idx2):
        if valid[i] and valid[j] and i != j:
            if np.random.rand() > 0.5:
                valid[i] = 0
            else:
                valid[j] = 0
    valid_idx = np.where(valid==1)
    return ph[valid_idx], pw[valid_idx]

def remove_border(mask):
        mask[0,:] = 0
        mask[:,0] = 0
        mask[mask.shape[0]-1,:] = 0
        mask[:,mask.shape[1]-1] = 0

def flow_sampler(flow, strategy=['grid'], bg_ratio=1./6400, nms_ks=15, max_num_guide=-1, guidepoint=None):
    assert bg_ratio >= 0 and bg_ratio <= 1, "sampling ratio must be in (0, 1]"
    for s in strategy:
        assert s in ['grid', 'uniform', 'gradnms', 'watershed', 'single', 'full', 'specified'], "No such strategy: {}".format(s)
    h = flow.shape[0]
    w = flow.shape[1]
    ds = max(1, max(h, w) // 400) # reduce computation

    if 'full' in strategy:
        sparse = flow.copy()
        mask = np.ones(flow.shape, dtype=np.int)
        return sparse, mask

    pts_h = []
    pts_w = []
    if 'grid' in strategy:
        stride = int(np.sqrt(1./bg_ratio))
        mesh_start_h = int((h - h // stride * stride) / 2)
        mesh_start_w = int((w - w // stride * stride) / 2)
        mesh = np.meshgrid(np.arange(mesh_start_h, h, stride), np.arange(mesh_start_w, w, stride))
        pts_h.append(mesh[0].flat)
        pts_w.append(mesh[1].flat)
    if 'uniform' in strategy:
        pts_h.append(np.random.randint(0, h, int(bg_ratio * h * w)))
        pts_w.append(np.random.randint(0, w, int(bg_ratio * h * w)))
    if "gradnms" in strategy:
        ks = w // ds // 20
        edge = get_edge(flow[::ds,::ds,:])
        kernel = np.ones((ks, ks), dtype=np.float32) / (ks * ks)
        subkernel = np.ones((ks//2, ks//2), dtype=np.float32) / (ks//2 * ks//2)
        score = signal.convolve2d(edge, kernel, boundary='symm', mode='same')
        subscore = signal.convolve2d(edge, subkernel, boundary='symm', mode='same')
        score = score / score.max() - subscore / subscore.max()
        nms_res = nms(score, nms_ks)
        pth, ptw = np.where(nms_res > 0.1)
        pts_h.append(pth * ds)
        pts_w.append(ptw * ds)
    if "watershed" in strategy:
        edge = get_edge(flow[::ds,::ds,:])
        edge /= max(edge.max(), 0.01)
        edge = (edge > 0.1).astype(np.float32)
        watershed = ndimage.distance_transform_edt(1-edge)
        nms_res = nms(watershed, nms_ks)
        remove_border(nms_res)
        pth, ptw = np.where(nms_res > 0)
        pth, ptw = neighbor_elim(pth, ptw, (nms_ks-1)/2)
        pts_h.append(pth * ds)
        pts_w.append(ptw * ds)
    if "single" in strategy:
        pth, ptw = np.where((flow[:,:,0] != 0) | (flow[:,:,1] != 0))
        randidx = np.random.randint(len(pth))
        pts_h.append(pth[randidx:randidx+1])
        pts_w.append(ptw[randidx:randidx+1])
    if 'specified' in strategy:
        assert guidepoint is not None, "if using \"specified\", switch \"with_info\" on."
        pts_h.append(guidepoint[:,1])
        pts_w.append(guidepoint[:,0])

    pts_h = np.concatenate(pts_h)
    pts_w = np.concatenate(pts_w)

    if max_num_guide == -1:
        max_num_guide = np.inf

    randsel = np.random.permutation(len(pts_h))[:len(pts_h)]
    selidx = randsel[np.arange(min(max_num_guide, len(randsel)))]
    pts_h = pts_h[selidx]
    pts_w = pts_w[selidx]

    sparse = np.zeros(flow.shape, dtype=flow.dtype)
    mask = np.zeros(flow.shape, dtype=np.int)
    
    sparse[:, :, 0][(pts_h, pts_w)] = flow[:, :, 0][(pts_h, pts_w)]
    sparse[:, :, 1][(pts_h, pts_w)] = flow[:, :, 1][(pts_h, pts_w)]
    
    mask[:,:,0][(pts_h, pts_w)] = 1
    mask[:,:,1][(pts_h, pts_w)] = 1
    return sparse, mask

def image_flow_aug(img1, img2, flow, flip_horizon=True):
    if flip_horizon:
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            flow = flow[:,::-1,:].copy()
            flow[:,:,0] = -flow[:,:,0]
    return img1, img2, flow

def flow_aug(flow, reverse=True, scale=True, rotate=True):
    if reverse:
        if random.random() < 0.5:
            flow = -flow
    if scale:
        rand_scale = random.uniform(0.5, 2.0)
        flow = flow * rand_scale
    if rotate and random.random() < 0.5:
        lengh = np.sqrt(np.square(flow[:,:,0]) + np.square(flow[:,:,1]))
        alpha = np.arctan(flow[:,:,1] / flow[:,:,0])
        theta = random.uniform(0, np.pi*2)
        flow[:,:,0] = lengh * np.cos(alpha + theta)
        flow[:,:,1] = lengh * np.sin(alpha + theta)
    return flow

def draw_gaussian(img, pt, sigma, type='Gaussian'):
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


