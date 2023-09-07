#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
import pickle
import lzma
import numpy as np
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

from typing import List

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--save_media",
        default=False,
        action="store_true",
        help="whether to save the inference visual result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--compress",
        default=False,
        action="store_true",
        help="whether or not to compress prediction output file",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        type=str,
        help="Path to TensorRT model",
    )
    parser.add_argument(
        "--fps",
        default=15,
        type=int,
        help="FPS of output video.",
    )
    parser.add_argument(
        "--batch",
        default=1,
        type=int,
        help="Batch inference size.",
    )
    parser.add_argument(
        "--out_postfix",
        type=str,
        help="postfix of output file basename",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            if self.fp16:
                x = x.half()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}

        height, width = img.shape[:2]
        if len(img.shape) == 4:
            height, width = img.shape[1:3]
        img_info["height"] = height
        img_info["width"] = width

        ratio = min(self.test_size[0] / height, self.test_size[1] / width)
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        return outputs, img_info

    def visual(self, output, img, img_info, cls_conf=0.35):
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]
        scores = output[:, 4]
        cls = output[:, 5]

        bboxes[:, 0] *= img_info["width"]
        bboxes[:, 1] *= img_info["height"]
        bboxes[:, 2] *= img_info["width"]
        bboxes[:, 3] *= img_info["height"]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, args):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        img = cv2.imread(image_name)
        img_tensor = torch.from_numpy(img).unsqueeze(0).cuda().half()
        outputs, img_info = predictor.inference(img_tensor)
        if len(outputs) <= 0:
            continue
        result_image = predictor.visual(outputs[0], img, img_info, predictor.confthre)
        if args.save_media or args.save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            if args.save_media:
                cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = args.fps if args.demo == "video" else cap.get(cv2.CAP_PROP_FPS)
    vid_writer = None
    if args.save_media or args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        if args.save_media:
            logger.info(f"video save_path is {save_path}")
            vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
            )
    frame_cnt = 0
    num_output_floats: int = 0
    all_outputs: List[torch.Tesnor] = []
    while True:
        buf: List[torch.Tensor] = list()
        ret_val = True
        i = 0
        while ret_val and i < args.batch:
            ret_val, frame = cap.read()
            if ret_val:
                buf.append(torch.from_numpy(frame).unsqueeze(0))
                i += 1
                frame_cnt += 1
        if len(buf) <= 0:
            break
        frame_batch = torch.cat(buf, dim=0)
        if frame_batch.shape[0] < args.batch:
            frame_batch = torch.cat([frame_batch, torch.zeros((
                args.batch - frame_batch.shape[0],
                frame_batch.shape[1],
                frame_batch.shape[2],
                frame_batch.shape[3],
            ))], dim=0)
        frame_batch = frame_batch.to(f"cuda:{torch.cuda.current_device()}")
        outputs, img_info = predictor.inference(frame_batch)
        outputs = outputs[:len(buf)]
        for j, cur_output in enumerate(outputs):
            cur_output[:, :4] /= img_info["ratio"] # scale boxes by ratio
            cur_output[:, 4] *= cur_output[:, 5] # calculate scores
            cur_output[:, 5] = cur_output[:, 6] # move pred cls left one column
            cur_output = cur_output[:, :6] # remove last column
            cur_output[:, 0] /= img_info["width"] # Change x-axis top-left anchor to ratio
            cur_output[:, 1] /= img_info["height"] # Change y-axis top-left anchor to ratio
            cur_output[:, 2] /= img_info["width"] # Change width to ratio
            cur_output[:, 3] /= img_info["height"] # Change height to ratio
            if args.save_result:
                all_outputs.append(cur_output.cpu().half().numpy())
            if len(cur_output) <= 0:
                continue
            num_output_floats += torch.numel(cur_output)
            if vid_writer is not None:
                result_frame = predictor.visual(cur_output, frame_batch[j], img_info, predictor.confthre)
                try:
                    vid_writer.write(result_frame)
                except Exception:
                    pass
        batch_step = ((1023 // args.batch) + 1) * args.batch
        if frame_cnt % batch_step == 0:
            time_fmt = "%Y-%m-%dT%H:%M:%S"
            print(f"{time.strftime(time_fmt, time.localtime())}: processed frame {frame_cnt}")
    print(f"Number of floats in outputs: {num_output_floats}")
    assert frame_cnt == len(all_outputs)
    if args.save_result:
        data: bytes = pickle.dumps(all_outputs)
        base_names: List[str] = os.path.basename(args.path).split(".")
        if len(base_names) > 1:
            base_names = base_names[:-1]
        if args.out_postfix:
            base_names[0] += f"_{args.out_postfix}"
        base_names.append("pkl")
        if args.compress:
            lzc = lzma.LZMACompressor()
            out1: bytes = lzc.compress(data)
            out2: bytes = lzc.flush()
            data = b"".join([out1, out2])
            base_names.append("lzma")
        out_basename = ".".join(base_names)
        res_path = os.path.join(save_folder, out_basename)
        with open(res_path, "wb") as outfile:
            outfile.write(data)

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result or args.save_media:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        assert os.path.exists(
            args.trt
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, args.trt, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
