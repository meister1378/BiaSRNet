# BIASRNet

BIASRNet (Burst Intermediate Alignment Super-Resolution Net): multi-frame
(burst) thermal video super-resolution, x4.

RRDB feature extraction + EDVR-style PCD (Pyramid, Cascading,
Deformable-conv) alignment of every neighbor frame's features onto the
center frame's features + RRDB body + pixel-shuffle upsampling, with the
center frame's bicubic upsample kept as a long residual skip. Optimized for
PSNR. See `model.py` for the full architecture notes and references.

A diffusion-based perceptual-quality refinement stage built on top of a
frozen BIASRNet checkpoint is available as a separate project, BIADSRNet.

## Files

- `model.py` -- BIASRNet + PCD alignment module
- `align.py` -- classical Farneback pre-alignment for a real LR burst
- `infer.py` -- runs a trained BIASRNet on a real LR burst (no synthetic
  degradation -- this is the production inference path)

This repo ships inference code only. You'll need a trained checkpoint
(`.pth`) to run it.

## Inference

```bash
python infer.py --frames-dir /path/to/real_lr_frames \
    --ckpt ./checkpoints/biasrnet_bestPSNR.pth --out-dir ./out
```

`--frames-dir` should contain real, already-low-resolution consecutive
grayscale frames.

## Requirements

```
torch
torchvision
opencv-python
numpy
```

## Data

Trained and evaluated on two public, permissively-licensed real thermal
video sources:
- [IRVAL](https://huggingface.co/datasets/zytlhy/IRVAL) (CC-BY-4.0)
- [issai/OpenThermalAction](https://huggingface.co/datasets/issai/OpenThermalAction) (MIT)

No other dataset is used or redistributed by this code.
