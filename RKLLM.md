# RKLLM 过程文档



## 1. 下载 RKNN-LLM

在虚拟机空间充足的目录下，通过git拉取RKNN-LLM

```shell
$ git clone https://github.com/airockchip/rknn-llm.git
$ ls rknn-llm/
CHANGELOG.md  doc  examples  LICENSE  README.md  res  rkllm-runtime  rkllm-toolkit  rknpu-driver  scripts
```

> doc中是RK的官方文档，可以参考官方文档，细节较多，下面介绍的主要还是操作过程

## 2. RKLLM-Toolkit环境搭建

### 2.1 安装Miniconda

为防止系统对多个不同版本的 Python 环境的需求，建议使用 miniforge3 管理 Python 环境。  

[MIniconda安装官网](https://www.anaconda.com/docs/getting-started/miniconda/install)

下载Miniconda3-latest-Linux-x86_64.sh

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

将Miniconda3-latest-Linux-x86_64.sh运行安装

```shell
$ bash Miniconda3-latest-Linux-x86_64.sh -b -u -p
```

### 2.2 RKLLM-Toolkit Conda 环境  

1. 安装后，进入 Conda base 环境  

   ```shell
   $ source miniconda3/bin/activate
   ```

2. 创建rkllm Conda环境

   ```shell
   $ conda create -n rkllm python=3.8
   .....
   Proceed ([y]/n)? 
   $ y
   ```

   > 这里指定的python版本为3.8，RK官方推荐有3.8 3.10两种，对应rknn-llm中的wheel包文件

3. 进入rkllm Conda环境

   ```shell
   (base) xie@xie:/home/work/rockchip/other$ conda activate rkllm
   (rkllm) xie@xie:/home/work/rockchip/other$ 
   ```

4. 拷贝rkllm_toolkit-1.1.4-cp38-cp38-linux_x86_64.whl

   ```shell
   $ cp rknn-llm/rkllm-toolkit/rkllm_toolkit-1.1.4-cp38-cp38-linux_x86_64.whl ./ -rf
   ```

5. 运行命令安装瑞芯微提供的rkllm_toolkit-1.0.0版本的软件包

   ```shell
   $ pip install rkllm_toolkit-1.1.4-cp38-cp38-linux_x86_64.whl
   ```

   > 等待下载安装完成 时间会比较久

6. 出现下面打印便是成功

```shell
Successfully built transformers-stream-generator
Installing collected packages: sentencepiece, pytz, mpmath, genson, flatbuffers, einops, zipp, xxhash, urllib3, tzdata, typing-extensions, tqdm, tomli, toml, tabulate, sympy, six, safetensors, rpds-py, regex, pyyaml, psutil, protobuf, propcache, platformdirs, pkgutil-resolve-name, pillow, pathspec, packaging, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, networkx, mypy-extensions, MarkupSafe, isort, inflect, idna, humanfriendly, fsspec, frozenlist, filelock, dnspython, dill, colorlog, click, charset-normalizer, certifi, attrs, async-timeout, argcomplete, aiohappyeyeballs, triton, rouge, requests, referencing, python-dateutil, pydantic-core, pyarrow, nvidia-cusparse-cu12, nvidia-cudnn-cu12, multiprocess, multidict, Jinja2, importlib-resources, gekko, email-validator, coloredlogs, black, annotated-types, aiosignal, yarl, tiktoken, pydantic, pandas, nvidia-cusolver-cu12, jsonschema-specifications, huggingface-hub, torch, tokenizers, jsonschema, aiohttp, transformers, torchvision, datamodel-code-generator, accelerate, transformers-stream-generator, peft, datasets, optimum, auto-gptq, rkllm-toolkit
Successfully installed Jinja2-3.1.4 MarkupSafe-2.1.5 accelerate-0.26.0 aiohappyeyeballs-2.4.4 aiohttp-3.10.11 aiosignal-1.3.1 annotated-types-0.7.0 argcomplete-3.6.1 async-timeout-5.0.1 attrs-25.3.0 auto-gptq-0.7.1 black-24.8.0 certifi-2025.1.31 charset-normalizer-3.4.1 click-8.1.8 coloredlogs-15.0.1 colorlog-6.8.2 datamodel-code-generator-0.26.0 datasets-2.14.6 dill-0.3.7 dnspython-2.6.1 einops-0.4.1 email-validator-2.2.0 filelock-3.16.1 flatbuffers-24.3.25 frozenlist-1.5.0 fsspec-2023.10.0 gekko-1.2.1 genson-1.3.0 huggingface-hub-0.29.3 humanfriendly-10.0 idna-3.10 importlib-resources-6.4.5 inflect-5.6.2 isort-5.13.2 jsonschema-4.23.0 jsonschema-specifications-2023.12.1 mpmath-1.3.0 multidict-6.1.0 multiprocess-0.70.15 mypy-extensions-1.0.0 networkx-3.1 numpy-1.23.1 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.18.1 nvidia-nvjitlink-cu12-12.8.93 nvidia-nvtx-cu12-12.1.105 optimum-1.23.3 packaging-24.2 pandas-2.0.3 pathspec-0.12.1 peft-0.13.2 pillow-10.4.0 pkgutil-resolve-name-1.3.10 platformdirs-4.3.6 propcache-0.2.0 protobuf-3.20.3 psutil-7.0.0 pyarrow-17.0.0 pydantic-2.10.6 pydantic-core-2.27.2 python-dateutil-2.9.0.post0 pytz-2025.2 pyyaml-6.0.2 referencing-0.35.1 regex-2024.11.6 requests-2.32.3 rkllm-toolkit-1.1.4 rouge-1.0.1 rpds-py-0.20.1 safetensors-0.4.2 sentencepiece-0.1.97 six-1.17.0 sympy-1.13.3 tabulate-0.9.0 tiktoken-0.4.0 tokenizers-0.20.3 toml-0.10.2 tomli-2.2.1 torch-2.1.0 torchvision-0.16.0 tqdm-4.64.1 transformers-4.45.0 transformers-stream-generator-0.0.4 triton-2.1.0 typing-extensions-4.13.0 tzdata-2025.2 urllib3-2.2.3 xxhash-3.5.0 yarl-1.15.2 zipp-3.20.2
```



## 3. DeepSeek大语言模型转换

### 3.1 下载DeepSeek模型

请根据你的板子实际内存，按需下载，不然也是运行不了的。

[DeepSeek网址](https://huggingface.co/deepseek-ai)

1.5B: 需4G内存板子

```shell
$ git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
$ git lfs install
$ git lfs pull
```

> 实际消耗约2.3G

7B: 需8G内存板子

```shell
$ git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
$ git lfs install
$ git lfs pull
```

> 运行消耗内存约6G

### 3.2 模型转换

拷贝脚本到同一目录方便操作

```shell
$ cp rknn-llm/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/export/* ./ -rf
```

运行脚本进行构建

```shell
$ python3 generate_data_quant.py -m DeepSeek-R1-Distill-Qwen-1.5B
```

修改模型生成的路径

```shell
$ vim export_rkllm.py
modelpath = 'DeepSeek-R1-Distill-Qwen-1.5B'
```

默认demo是转换3588的

```shell
python3 export_rkllm.py
```

转换3576需要修改demo

```diff
+target_platform = "RK3576"
optimization_level = 1
+quantized_dtype = "w4a16_g32"  #w4a16_g64 or w4a16_g128
quantized_algorithm = "normal"
+num_npu_core = 2
```

> target_platform: 模型运行的硬件平台, 可选择的设置包括“rk3576”或“rk3588”；
>
> quantized_dtype: 目前rk3576 平台支持“ w4a16” ,“ w4a16_g32” ,“ w4a16_g64” ,“ w4a16_g128” 和“ w8a8” 五种量化类型， rk3588 支持“ w8a8” ,“ w8a8_g128” ,“ w8a8_g256” , “ w8a8_g512 ” 四 种 量 化 类 型 ；
>
> num_npu_core: 模型推理需要使用的 npu 核心数， “rk3576”可选项为[1,2]， “rk3588”可选项为[1,2,3]；



## 4. 推理demo程序的交叉编译

进入rknn-llm工具包的这个目录

```shell
$ cd rknn-llm/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/deploy
$ ls
build  build-android.sh  build-linux.sh  CMakeLists.txt  install  src
```

修改对应脚本的内容

### 4.1 Linux系统

使用build-linux.sh

主要修改下面这个定义的路径为你的交叉编译工具链

```
GCC_COMPILER_PATH=/home/work/rockchip/rk3576-linux/prebuilts/gcc/linux-x86/aarch64/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu
```

编译后会生成install目录

```
$ ls install/demo_Linux_aarch64/
lib  llm_demo
```

> 这些便是我们要使用的demo

### 4.2 Android系统

使用build-android.sh

主要修改下面这个

```
ANDROID_NDK_PATH=~/opt/android-ndk-r24e
```



## 5. 运行测试

 将转换生成的模型和rknn-llm生成的install拷贝到板子里面

```shell
$ ls
DeepSeek-R1-Distill-Qwen-1.5B_w4a16_g32_RK3576.rkllm  demo_Linux_aarch64
```

运行下面命令进行测试

```shell
$ cd demo_Linux_aarch64
$ export LD_LIBRARY_PATH=./lib
$ ./llm_demo ../DeepSeek-R1-Distill-Qwen-1.5B_w4a16_g32_RK3576.rkllm 1024 2048
```

> 最后两个参数的介绍：
>
> 1024 ： 它定义了模型在生成文本时最多可以生成的新 token 数量。即模型输出的长度
>
> 2048： 它定义了模型在处理输入上下文时的最大上下文长度（以 token 为单位）。上下文长度包括输入 prompt 的 token 数量和生成的新 token 数量。这个你输入同一个问题 在不同参数下，这个参数越低回答越简单，甚至运行失败。

若需要查看 RKLLM 在板端推理的性能，可使用如下指令：  

```shell
$ export RKLLM_LOG_LEVEL=1
```

打开另一个终端，在root权限下查看运行过程中NPU的使用情况

```shell
# cat /sys/kernel/debug/rknup/load
```



