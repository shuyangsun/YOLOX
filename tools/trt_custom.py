#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import shutil
from loguru import logger

import torch
import tensorrt as trt
import torch_tensorrt

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
    parser.add_argument(
        "-w", '--workspace', type=int, default=32, help='max workspace size in detect'
    )
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
    model = model.eval().to('cuda:0')
    model_trt = torch_tensorrt.compile(model,
        # require_full_compilation = True,
        inputs = [
            torch_tensorrt.Input( # Specify input object with shape and dtype
                shape=[1, 3, exp.test_size[0], exp.test_size[1]],
                dtype=torch.float32,
            ),
        ],

        # For inputs containing tuples or lists of tensors, use the `input_signature` argument:
        # Below, we have an input consisting of a Tuple of two Tensors (Tuple[Tensor, Tensor])
        # input_signature = ( (torch_tensorrt.Input(shape=[1, 3, 224, 224], dtype=torch.half),
        #                      torch_tensorrt.Input(shape=[1, 3, 224, 224], dtype=torch.half)), ),

        enabled_precisions = {torch.float32}, # Run with FP16
    )
    torch.save(model_trt.state_dict(), os.path.join(file_name, "model_trt.pth"))
    logger.info("Converted TensorRT model done.")
    engine_file = os.path.join(file_name, "model_trt.engine")
    engine_file_demo = os.path.join("demo", "TensorRT", "cpp", "model_trt.engine")
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())

    shutil.copyfile(engine_file, engine_file_demo)

    logger.info("Converted TensorRT model engine file is saved for C++ inference.")


if __name__ == "__main__":
    main()
