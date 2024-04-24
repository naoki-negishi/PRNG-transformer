# ベースイメージの場所を指定
FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN echo "start building docker container!"
####
# install apt packages
####
# apt でのインストール時に timezone の選択が要求されないように事前に設定しておく
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
# apt パッケージのインストール
RUN apt-get -y update \
&& apt-get install build-essential libbz2-dev libdb-dev \
    libreadline-dev libffi-dev libgdbm-dev liblzma-dev \
    libncursesw5-dev libsqlite3-dev libssl-dev \
    zlib1g-dev uuid-dev tk-dev locales-all -y\
&& apt-get install -y zsh \
&& apt-get install -y vim \
&& apt-get install -y git \
&& apt-get install -y tmux \
&& apt-get install -y curl \
&& apt-get install -y wget \
&& apt-get install -y rsync \
&& apt-get install -y tree \
&& apt-get install -y lsb-release
####
# Define Environment Variables
####
# 環境変数を定義する
ENV HOME="/workspace"
ENV LANG="en_US.UTF-8"
# add path to use poetry
ENV PATH="/workspace/.local/bin:$PATH"
# 鈴木研サーバーを使用する場合、コメントアウトを外してください
## wandb directory
# ENV WANDB_DIR="/workspace/nlp-docker/"
# ENV WANDB_CONFIG_DIR="/workspace/nlp-docker/"
# ENV WANDB_CACHE_DIR="/workspace/nlp-docker/"
## huggingface transformers directory
# ENV HF_HOME=/work00/share/cache/huggingface

COPY entrypoint.sh /workspace/nlp-docker/entrypoint.sh
# コンテナ内の作業ディレクトリ(ログイン時のディレクトリ)の指定
WORKDIR /workspace
ENTRYPOINT ["zsh", "/workspace/nlp-docker/entrypoint.sh"]
