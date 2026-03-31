基于llava的文档识别助手
### 基于 LLaVA 的图表逻辑推理与计算智能体

#### 简介

项目主要通过在垂直领域数据集**HuggingFaceM4/ChartQA**（图表问答信息库）的基础上，通过**llama-factory**，微调**llava-hf/llava-1.5-7b-hf**模型，随后通过**llama.cpp**(github开源项目名称)，完成safetensors格式到gguf格式转换，并通过**llama.cpp**的量化工具完成量化，最后通过**phidata**（开源项目名称）实现智能rag，最后做可视化部署。

### 一.下载数据集 && 转换为llava模型微调的sharegpt格式

```
pip install huggingface-hub	##通过huggingface-hub下载数据集

hf download HuggingFaceM4/ChartQA --repo-type dataset --lacaldir /root/datasets #下载数据集到/root/datasets

python preprocess.py #注意指定其中的数据集路径后输出路径后运行
```

### 二.用llama-factory微调

安装过程可以去github搜索llama-factory项目([LlamaFactory/README_zh.md at main · hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory/blob/main/README_zh.md#安装-llama-factory))，安装完成后，运行llamafactory-cli webui，注意必须进入LlamaFactory 下启动可视化界面，同时，注册数据集完成微调



![image-20260330171453506](assets\image-20260330171453506.png)







![image-20260330171520779](assets\image-20260330171520779.png)

### 三.使用llama.cpp完成格式转换、量化部署

下载项目[ggml-org/llama.cpp: LLM inference in C/C++](https://github.com/ggml-org/llama.cpp/tree/master)

安装依赖并编译生成可执行文件

```
#进入llama.cpp文件夹后,安装依赖
pip install -r requeirements 

#执行编译文件
#注意llava是多模态模型，需要视觉部分切分出来才能完成转换，详情项目介绍
python3 convert_hf_to_gguf.py   输入文件路径   输出问价按路径   格式

#量化语言模型，视觉模型很小，不需要量化
llama-quantize 模型路径      输出路径    量化格式

#部署模型api，路径替换成自己的
./llama-server -m /root/merge_model/llava_llm_q4_k_m.gguf -mm /root/merge_model/mmproj-model-f16.gguf 
```

### 四.rag和可视化

首先，下载ollama拉取知识库嵌入模型**nomic-embed-text**，然后执行命令就可以对话了

```
pip install phidata gradio pypdf lancedb ollama
python rag.py
```

![image-20260330184534023](assets\image-20260330184534023.png)
