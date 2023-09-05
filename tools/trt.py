#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import cv2
import shutil
import time
from loguru import logger

import tensorrt as trt
import numpy as np
import torch
from torch2trt import torch2trt

from yolox.exp import get_exp
from typing import List, Tuple

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def prepare_samples(img_dir: str, input_size: Tuple[int, int]) -> torch.Tensor:
    res: List[torch.Tensor] = list()
    for root, _, files in os.walk(img_dir):
        for f in files:
            f_lower = f.lower()
            if not f_lower.endswith(".png") or f_lower.endswith(".jpg") or f_lower.endswith(".jpeg"):
                continue
            img = cv2.imread(os.path.join(root, f))
            img, _ = preproc(img, input_size)
            img = torch.from_numpy(img).unsqueeze(0).half().to('cuda:0')
            res.append(img)
            break
    return torch.cat(res)

def make_parser():
    parser = argparse.ArgumentParser("YOLOX ncnn deploy")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "-w", '--workspace', type=int, default=32, help='max workspace size in detect'
    )
    parser.add_argument("-b", '--batch', type=int, default=1, help='max batch size in detect')
    parser.add_argument("-i", '--inputs', type=str, help='List of input dir for sample inputs.')
    return parser


@logger.catch
@torch.no_grad()
def main():
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)
    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict

    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    model.head.decode_in_inference = False
    model = model.eval().half().to('cuda:0')
    inputs = [torch.ones(1, 3, exp.test_size[0], exp.test_size[1], dtype=torch.float16, device="cuda:0")]
    if args.inputs is not None:
        inputs = [prepare_samples(args.inputs, exp.test_size)]

    num_samples = inputs[0].shape[0]
    num_iter = 150
    start = time.time()
    for _ in range(num_iter):
        for i in range(num_samples):
            _ = model(inputs[0][i:i+1])
    print("PyTorch model fps (avg of {num} samples): {fps:.3f}".format(num=num_samples, fps=(time.time() - start)/(num_samples * num_iter)))

    model_trt = torch2trt(
        model,
        inputs,
        fp16_mode=True,
        log_level=trt.Logger.INFO,
        max_workspace_size=(1 << args.workspace),
        max_batch_size=args.batch,
    )

    start = time.time()
    for _ in range(num_iter):
        for i in range(num_samples):
            _ = model_trt(inputs[0][i:i+1])
    print("TensorRT model fps (avg of {num} samples): {fps:.3f}".format(num=num_samples, fps=(time.time() - start)/num_iter))

    basename = os.path.basename(args.ckpt)
    components = basename.split(".")[:-1]
    components[-1] += "_trt"
    out_basename = ".".join(components)

    torch.save(model_trt.state_dict(), os.path.join(file_name, f"{out_basename}.pth"))
    logger.info("Converted TensorRT model done.")
    engine_file = os.path.join(file_name, f"{out_basename}.engine")
    engine_file_demo = os.path.join("demo", "TensorRT", "cpp", f"{out_basename}.engine")
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())

    shutil.copyfile(engine_file, engine_file_demo)

    logger.info("Converted TensorRT model engine file is saved for C++ inference.")


if __name__ == "__main__":
    main()
