# nlp-docker

Dockerを使ってNLP研究をするためのひな型レポジトリ。<br>
有名なパッケージやライブラリは予めインストールされるようにしてあります。<br>
Python の実行環境構築には、[Pyenv](https://github.com/pyenv/pyenv) + [venv](https://docs.python.org/ja/3/library/venv.html) + [Poetry](https://github.com/python-poetry/poetry) を採用しています。<br>

>- **[Pyenv](https://github.com/pyenv/pyenv)**<br>
複数のバージョンの Python を簡単にインストールでき、プロジェクトごとに切り替えたりすることができます。
>- **[venv](https://docs.python.org/ja/3/library/venv.html)**<br>
Python の仮想環境を構築することができます。
>- **[Poetry](https://github.com/python-poetry/poetry)**<br>
外部パッケージ同士の依存関係の管理を行ってくれます。

**本リポジトリで紹介する Python 実行環境 では、これらのツールを使用することで、以下のことが実現できます。**

- **複数のバージョンの Python を簡単にインストールでき、切り替えることができる**
- **プロジェクトごとに必要な外部パッケージをまとめて切り替えることができる**
- **外部パッケージ同士の依存関係を管理し、破壊的なパッケージの変更を行えないようにすることができる**
- **構築した環境に再現性を持たせることができる**

>より詳しい採用理由に関しては、[研究のためのPython開発環境 Python環境構築[Pyenv+Poetry]](https://zenn.dev/zenizeni/books/a64578f98450c2/viewer/c6af80#%E6%8E%A1%E7%94%A8%E7%90%86%E7%94%B1(%E5%80%8B%E4%BA%BA%E7%9A%84%E6%84%9F%E6%83%B3%E3%82%92%E5%90%AB%E3%81%BF%E3%81%BE%E3%81%99)) に書かれていることとほとんど同じです。<br>
>詳しいことが知りたい方は、参照することをおすすめします。

主要なソフトウェアのバージョンは以下の通りです。

![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04.2LTS-red)
![CUDA](https://img.shields.io/badge/CUDA-v11.8-red)
![Python](https://img.shields.io/badge/Python-v3.10.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0.1-blue)
![pyenv](https://img.shields.io/badge/pyenv-v2.3.24-green)
![poetry](https://img.shields.io/badge/poetry-v1.5.1-green)

> Singularity版はこちら：<br>
> https://github.com/cl-tohoku/nlp-singularity

## 環境構築

container環境を構築するためには、dockerが利用可能な任意のサーバー内に入り、以下のコマンドを順に実行してください。

### pyenv の PATH の指定

`.zshrc`ファイルに以下を追記して下さい。

```shell
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
export PATH="$HOME/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
```

```shell
source $HOME/.zshrc
```

### container のビルドとpyhton実行環境の構築

container環境を構築するためには、Dockerが利用可能な任意のサーバー内に入り、以下のコマンドを順に実行してください。

```shell
cd ~/

git clone https://github.com/cl-tohoku/nlp-docker.git
cd nlp-docker

# write your username to .env file
echo "HOME=$HOME" > .env
echo "HOST_USERNAME=$(id -un)" >> .env

# huggingface の default model cache を変えたい場合、Dockerfile の以下のコメントアウトを外してください
vim Dockerfile
# 鈴木研サーバーを使用する場合、コメントアウトを外してください
## wandb directory
# ENV WANDB_DIR="/workspace/nlp-docker/"
# ENV WANDB_CONFIG_DIR="/workspace/nlp-docker/"
# ENV WANDB_CACHE_DIR="/workspace/nlp-docker/"
## huggingface transformers directory
# ENV HF_HOME=/work00/share/cache/huggingface
## (注意): huggingface の default model cache を変えた場合、bind mount を適切に設定するようにしてください

# make container image
docker compose build

# create container
docker compose up -d

# attach to running container
# コンテナの名前は適宜 docker ps で確認できます
docker exec -it nlp-docker-nlp-1 /bin/zsh

# これ以降はコンテナ内部にいる想定です
# build pyenv + poetry environment (4 min)
cd /workspace/nlp-docker
sh setting.sh

cd /workspace/nlp-docker/project

# 構築した環境の情報を取得 (optional)
## 仮想環境で使用するpythonのversionの確認
poetry env info
## 仮想環境でインストールしたpythonパッケージのversionの確認
poetry show
```

## 動作確認 (Optional)

ここでは、動作確認と [Pyenv](https://github.com/pyenv/pyenv) + [venv](https://docs.python.org/ja/3/library/venv.html) + [Poetry](https://github.com/python-poetry/poetry) を用いた実験の流れを BERT ベースの文書分類モデルの学習を通して確認していきます。<br>
また、

- 実験設定の管理：[Hydra](https://hydra.cc/docs/intro/)
- 実験結果の管理：[Weights & Biases (W&B)](https://wandb.ai/site)

を使用しています。<br>
実験設定の確認・変更は、`playground/configs`下のファイルを参照・編集して下さい。

```shell
# プロジェクトディレクトリに移動
cd /workspace/nlp-docker/playground/code

pyenv local 3.10.12
# 仮想環境の構築 (poetry をダウンロードした python と別のバージョンを使用する場合必須)
python -m venv .venv
# pyproject.toml, poetry.lock をもとに module をインストール
poetry install

# 前処理を行うスクリプトを実行
# (今回はこのスクリプトを実行することで生成されるデータファイルが、
#  git clone 時に playground/data 下にダウンロードされているので、実行しなくても良い)
poetry run python preprocess.py

# 学習を行うスクリプトを実行
poetry run python train.py
```

checkpointは、`playground/outputs`下に保存されているはずです。<br>
また、Weights & Biases (W&B) を確認して、実験結果が記録されていたら OK です！<br>

>今回は、動作確認がメインなので、`playground/configs`下のファイル中の`dry_run`オプションを`True`にしたことで、理想的な学習ログが得られていないかと思います。<br>
>`dry_run`オプションを`True`にすることで、使用するデータ量とepoch数が小さくなるように実装してあります。

## Misc.
### 追加でライブラリ・パッケージをインストールしたい時

- 任意の Python ライブラリを追加したい場合は、`pyproject.toml` に追記する、もしくは、`＄ poetry add <package name>` で追加してください。

#### **Poetry で 別の CUDA版 PyTorch をインストールするときの注意点**

本リポジトリの `pyproject.toml` を参照して、Python の環境構築を行った場合、インストールされる PyTorch は CUDA Version: 11.8 に対応したものですが、場合によっては、別の CUDA Version に対応した PyTorch が欲しい状況もあるかと思います。<br>
そうした場合に、Poetry を使用して PyTorch のインストールをする方法は複数存在しているのですが、現状のベストプラクティスは以下の方法かと思われます。<br>
試しに、 CUDA Version: 11.7 に対応した PyTorch を Poetry でインストールする例を紹介します。
```shell
# torch_cu117という命名は変更可能
poetry source add torch_cu117 --priority=explicit https://download.pytorch.org/whl/cu117

poetry add torch torchvision torchaudio --source torch_cu117
```

このように指定することで、Poetry は指定した source (`torch_cu117`) を探しに行って、環境に沿った wheel ファイルを見つけて、インストールしてくれます。

poetry 1.5系以前は `--priority=explicit` オプションがなく、PyPIにあるライブラリをインストールする際も、追加したソースを含めて検索する選択肢しか提供されていないという問題を抱えていました。<br>
この仕様により、インストール時間が非常に長引いたり、ログが汚染されてしまうという、耐え難い挙動の変化が起きてしまっていたそうです（参考：[Poetry1.5.1からGPU版のPytorchのインストールが簡単になりました (Zenn)](https://zenn.dev/zerebom/articles/b338784c8ac76a)）。<br>
幸い、この問題は、poetry 1.5以降改善されたようなので、この方法を採用しています。

また、単純に PyTorch のバージョンだけを切り変えたい場合は、`pyproject.toml` の `version` の項目を修正してみてください。
```shell
torch = {version = "^2.0.1+cu118", source = "torch_cu118"}
torchvision = {version = "^0.15.2+cu118", source = "torch_cu118"}
torchaudio = {version = "^2.0.2+cu118", source = "torch_cu118"}
```

### Huggingface Transformers のモデルをパスを明示的に指定して保存し、これをオフラインで利用する

鈴木研サーバー上で、Huggingface Transformers のモデルをパスを明示的に指定して保存して、オフラインで利用する例を示します。<br>
また今回、モデルは、`/work00/share/cache/huggingface/hub` 下に保存することとします。

1. パスを明示的に指定して、モデルをダウンロードする (`playground/huggingface/download_hf_models.py`)<br><br>
`cache_dir` にモデルを保存したいパスを指定しましょう。

```python
# download huggingface model
from huggingface_hub import snapshot_download
from huggingface_hub import login
login()

# 1例として、`meta-llama/Llama-2-7b-hf` をダウンロードする
## Dockerfile で
## - ENV HF_HOME=/work00/share/cache/huggingface や
## - ENV TRANSFORMERS_CACHE=/work00/share/cache/huggingface/hub
## を指定している場合は、cache_dir を指定する必要はありません
snapshot_download(repo_id="meta-llama/Llama-2-7b-hf", cache_dir="/work00/share/cache/huggingface/hub")
```

2. モデルをオフラインで利用する (`playground/huggingface/hf_causal_inference.py`)<br><br>
`cache_dir` に先ほどモデルを保存したパスを指定しましょう。<br>
また、`local_files_only` を指定することで、Huggingface Hub にアクセスせずに、ローカルにあるファイルだけを探しに行ってくれます。<br>
（また、すでにローカルにモデルを保存してあったとしても、`local_files_only` を `True` に指定しないと Huggingface Hub にアクセスしてしまう仕様のようなので、この挙動は把握しておくと良いかもしれません）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
set_seed(42)

prompt = "What sightseeing spots do you recommend in Tokyo?"

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    cache_dir="/work00/share/cache/huggingface/hub",  # HFモデルを保存したパスを指定
    local_files_only=True  # ローカルにあるファイルだけを探しにいくようにする
    )
inputs = tokenizer(prompt, return_tensors="pt").input_ids

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    cache_dir="/work00/share/cache/huggingface/hub",
    local_files_only=True
    )
outputs = model.generate(
    inputs,
    num_beams=3,
    max_new_tokens=50,
    do_sample=True,
    top_k=0,
    top_p=0.5,
    temperature=1.3,
    no_repeat_ngram_size=3,
    )
output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(f"(user): {prompt}")
print("="*30)
print(f"(assistant): {output_text[0][len(prompt)+1:]}")
```

### 参考

- **環境構築**
    - [研究のためのPython開発環境 (Zenn)](https://zenn.dev/zenizeni/books/a64578f98450c2)
        - 非常によくまとまっていて、一度すべて目を通してみてほしい。本リポジトリを作成する上でも参考にしたところはとても多い。
    - [Poetry1.5.1からGPU版のPytorchのインストールが簡単になりました (Zenn)](https://zenn.dev/zerebom/articles/b338784c8ac76a)
