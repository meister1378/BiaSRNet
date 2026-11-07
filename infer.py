"""Run a trained BIASRNet on a real low-resolution burst (no synthetic
degradation -- this is the production inference path, for frames that are
already your sensor's native LR output).

Takes a directory of >= n_frames consecutive grayscale LR frames, aligns the
neighbors onto the center frame with Farneback optical flow (same
pre-alignment used at training time), runs BIASRNet, and writes the restored
x4 image (plus a bicubic baseline for reference).

Usage:
    python infer.py --frames-dir /path/to/lr_frames --center 7 --ckpt ./checkpoints/biasrnet_bestPSNR.pth --out-dir ./out
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from align import align_to_center
from model import BIASRNet


def load_frames(frames_dir):
    files = sorted(os.listdir(frames_dir))
    return [cv2.imread(os.path.join(frames_dir, f), cv2.IMREAD_GRAYSCALE) for f in files]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames-dir', required=True, help='directory of consecutive grayscale LR frames')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--center', type=int, default=None,
                   help='index (into the sorted frame list) of the frame to restore; default = middle frame')
    p.add_argument('--n-frames', type=int, default=5)
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--out-dir', default='./out')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    frames = load_frames(args.frames_dir)
    half = args.n_frames // 2
    center = args.center if args.center is not None else len(frames) // 2
    if center - half < 0 or center + half >= len(frames):
        raise ValueError(f'center={center} needs {half} frames of margin on each side '
                          f'(have {len(frames)} frames)')
    indices = list(range(center - half, center + half + 1))
    window = [np.ascontiguousarray(frames[i]) for i in indices]

    aligned = align_to_center(window, half)
    burst = np.stack(aligned, axis=0).astype(np.float32) / 255.0
    burst_t = torch.from_numpy(burst).float().unsqueeze(0).to(device)

    model = BIASRNet(n_frames=args.n_frames, deform_groups=1)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()

    with torch.no_grad():
        pred = model(burst_t).clamp(0, 1)
        center_t = burst_t[:, half:half + 1]
        bicubic = F.interpolate(center_t, scale_factor=args.scale, mode='bicubic', align_corners=False)

    pred8 = (pred.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    bicubic8 = (bicubic.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(args.out_dir, 'restored_biasrnet.png'), pred8)
    cv2.imwrite(os.path.join(args.out_dir, 'bicubic_baseline.png'), bicubic8)
    print(f'wrote restored_biasrnet.png + bicubic_baseline.png to {args.out_dir} (center frame index {center})')


if __name__ == '__main__':
    main()
