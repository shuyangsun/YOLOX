FROM nvcr.io/nvidia/tensorrt:23.11-py3

# Build image:
# Copy "yolox_l.pth" is under the "model" directory in this repo first.
# docker image build . -t ssml/yolox:latest

# Run image:
# docker run --gpus device=0 ssml/yolox:latest

# Setup Python

# YOLOX

RUN mkdir /yolox
WORKDIR /yolox

RUN apt update
RUN apt-get install python3.10-venv ffmpeg libsm6 libxext6 -y
RUN python -m venv .venv
RUN source .venv/bin/activate

ADD requirements_exact.txt requirements.txt
RUN pip install -r requirements.txt

ADD yolox yolox
ADD exps exps
ADD tools tools

RUN mkdir model

ADD model/yolox_l.pth model/yolox_l.pth

ENTRYPOINT python tools/trt.py --exp_file exps/default/yolox_x.py -c model/yolox_x.pth -i 1024 -b 128 -d cuda:0
