FROM nvcr.io/nvidia/tensorrt:23.11-py3

# Install NVIDIA container runtime following instructions at
# https://stackoverflow.com/a/61737404/2177724

# Copy "yolox_x.pth" is under the "model" directory in this repo first.

# Build image:
# docker image build . -t ssml/yolox:latest

# Run image:
# docker run --gpus device=0 ssml/yolox:latest

ENV CUDA_HOME=/usr/local/cuda
ENV BATCH 32

# YOLOX
RUN mkdir /yolox
WORKDIR /yolox

RUN apt update
RUN apt-get install python3.10-venv ffmpeg libsm6 libxext6 -y
RUN python -m venv .venv
RUN source .venv/bin/activate

ADD requirements_exact.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /
ADD torch2trt torch2trt
WORKDIR /torch2trt
RUN python setup.py install

WORKDIR /yolox
ADD yolox yolox
ADD exps exps
ADD tools tools
ADD setup.py setup.py
ADD README.md README.md
RUN pip install -e .

RUN mkdir model
ADD model/yolox_x.pth model/yolox_x.pth

WORKDIR /yolox
ENTRYPOINT python tools/trt.py --exp_file exps/default/yolox_x.py -c model/yolox_x.pth -i 1024 -b ${BATCH} -d cuda:0
