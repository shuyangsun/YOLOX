#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import cv2
import time
from loguru import logger

import torch
import torch_tensorrt
import numpy as np

from yolox.exp import get_exp


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
    parser.add_argument("-d", "--device", default="cuda:0", type=str, help="device")
    parser.add_argument(
        "-w", "--workspace", type=int, default=32, help="max workspace size in detect"
    )
    return parser


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


@logger.catch
@torch.no_grad()
def main():
    args = make_parser().parse_args()
    with torch.cuda.device(args.device):
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
        model = model.eval().half().to(args.device)
        img = cv2.imread("/home/ssun/Desktop/test_img.png")
        img, _ = preproc(img, exp.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.half().to(args.device)
        test_iter = 10
        start = time.time()
        for _ in range(test_iter):
            _ = model(img)
        print(f"torch: {(time.time() - start)/test_iter}")
        model_trt = torch_tensorrt.compile(
            model,
            # require_full_compilation = True,
            inputs=[
                img,
                # torch_tensorrt.Input( # Specify input object with shape and dtype
                #     shape=[1, 3, exp.test_size[0], exp.test_size[1]],
                #     dtype=torch.float32,
                # ),
            ],
            # For inputs containing tuples or lists of tensors, use the `input_signature` argument:
            # Below, we have an input consisting of a Tuple of two Tensors (Tuple[Tensor, Tensor])
            # input_signature = ( (torch_tensorrt.Input(shape=[1, 3, 224, 224], dtype=torch.half),
            #                      torch_tensorrt.Input(shape=[1, 3, 224, 224], dtype=torch.half)), ),
            enabled_precisions={torch.half},  # Run with FP16
        )
        torch.save(model_trt.state_dict(), os.path.join(file_name, "model_trt.pth"))
        logger.info("Converted TensorRT model done.")
        engine_file = os.path.join(file_name, "model_trt.ts")
        torch.jit.save(model_trt, engine_file)
        start = time.time()
        for _ in range(test_iter):
            _ = model_trt(img)
        print(f"tensorrt: {(time.time() - start)/test_iter}")

        logger.info("Converted TensorRT model engine file is saved for C++ inference.")


if __name__ == "__main__":
    main()
