# LLMs-Inference-Papers

![](https://img.shields.io/github/downloads/DefTruth/LLMs-Inference-Papers/total?color=ccf&label=downloads&logo=github&logoColor=lightgrey) ![](https://img.shields.io/github/stars/DefTruth/LLMs-Inference-Papers.svg?style=social)

## 🌟说明
持续更新：最近想整体地读一下LLMs推理优化相关的Papers，但发现各个博客文章介绍到的知识点涉及的论文比较分散，于是将自己关注的一些LLMs推理优化技术论文整理成册，便于自己阅读查找，也在这里分享出来。格式：PDF，带标签，可跳转。更多论文可见，[✅LLM推理论文列表](#paperlist)，欢迎star🌟👨‍💻~

## ✅PDF下载  

点击下载：

- [LLMs-Inference-Papers-v0.1.zip](https://github.com/DefTruth/LLMs-Inference-Papers/releases/download/v0.1/LLMs-Inference-Papers-v0.1.zip): LLMs入门+推理入门  
- [LLMs-Inference-Papers-v0.2.zip](https://github.com/DefTruth/LLMs-Inference-Papers/releases/download/v0.2/LLMs-Inference-Papers-v0.2.zip): 精简版，仅包含推理论文  

或命令行下载：

```bash
wget https://github.com/DefTruth/LLMs-Inference-Papers/releases/download/v0.1/LLMs-Inference-Papers-v0.1.zip
wget https://github.com/DefTruth/LLMs-Inference-Papers/releases/download/v0.2/LLMs-Inference-Papers-v0.2.zip
```

## 🎉PDF更新  

- [x] [LLMs-Inference-Papers-v0.1.pdf](https://github.com/DefTruth/LLMs-Inference-Papers/releases/download/v0.1/LLMs-Inference-Papers-v0.1.zip): LLMs入门，偏优化，600页PDF。涉及Transformer、BN、LN、MQA、FlashAttention、FlashAttention2、GLM、GLM-130B、GPT-3、GPT-3.5、GPT-4、LLaMA、LLaMA2、LoRA、QLoRA、P-Tuning V1、P-Tuning V2、RoPE、SmoothQuant、WINT8/4、Continuous Batching（动态插入）、FP8等。

<img width="1788" alt="LLMs-Inference-Papers-v0 1_For_Beginners" src="https://github.com/DefTruth/LLMs-Inference-Papers/assets/31974251/03fac365-87da-4c9d-909c-ea2fe457b127">

- [x] [LLMs-Inference-Papers-v0.2.pdf](https://github.com/DefTruth/LLMs-Inference-Papers/releases/download/v0.2/LLMs-Inference-Papers-v0.2.zip): LLMs推理优化论文（**精简版，仅包含推理优化论文**），286页PDF。包含ByteTransformer、FastServe、FlashAttention、FlashAttention-2、FlexGen、FP8、LLM.int8()、Tensor Core相关、PagedAttention、RoPE、SmoothQuant、SpecInfer、WINT8/4、Continuous Batching、ZeroQuant等。

<img width="1440" alt="v0 2" src="https://github.com/DefTruth/LLMs-Inference-Papers/assets/31974251/bb136842-8054-4599-8bfe-36c36f0e997f">

## 📁LLM推理论文列表

<div id="paperlist"></div>  

|Date|Title|Paper|Code|
|:---:|:---:|:---:|:---:|
|2022.10|[ByteTransformer] A High-Performance Transformer Boosted for Variable-Length Inputs|[[arxiv][pdf]](https://arxiv.org/pdf/2210.03052.pdf)|[[GitHub] [ByteTransformer]](https://github.com/bytedance/ByteTransformer) ![](https://img.shields.io/github/stars/bytedance/ByteTransformer.svg?style=social)|   
|2022.07|[Continuous Batching] Orca: A Distributed Serving System for Transformer-Based Generative Models |[[osdi22-yu][pdf]](https://www.usenix.org/system/files/osdi22-yu.pdf)|-|          
|2023.05|[FastServe] Fast Distributed Inference Serving for  Large Language Models |[[arxiv][pdf]](https://arxiv.org/pdf/2305.05920.pdf)|-|       
|2022.05|[FlashAttention] Fast and Memory-Efficient Exact Attention with IO-Awareness |[[arxiv][pdf]](https://arxiv.org/pdf/2205.14135.pdf)|[[GitHub][flash-attention]](https://github.com/Dao-AILab/flash-attention) ![](https://img.shields.io/github/stars/Dao-AILab/flash-attention.svg?style=social)|     
|2023.07|[FlashAttention-2] Faster Attention with Better Parallelism and Work Partitioning |[[arxiv][pdf]](https://arxiv.org/pdf/2307.08691.pdf)|[[GitHub][flash-attention]](https://github.com/Dao-AILab/flash-attention) ![](https://img.shields.io/github/stars/Dao-AILab/flash-attention.svg?style=social)|    
|2023.03|[FlexGen] High-Throughput Generative Inference of Large Language Models  with a Single GPU |[[arxiv][pdf]](https://arxiv.org/pdf/2303.06865.pdf)|[[GitHub][FlexGen]](https://github.com/FMInference/FlexGen) ![](https://img.shields.io/github/stars/FMInference/FlexGen.svg?style=social)|       
|2022.09|[FP8] FP8 FORMATS FOR DEEP LEARNING |[[arxiv][pdf]](https://arxiv.org/pdf/2209.05433.pdf)|-|   
|2022.08|[LLM.int8()] 8-bit Matrix Multiplication  for Transformers at Scale |[[arxiv][pdf]](https://arxiv.org/pdf/2208.07339.pdf)|[[GitHub][bitsandbytes]](https://github.com/timdettmers/bitsandbytes) ![](https://img.shields.io/github/stars/timdettmers/bitsandbytes.svg?style=social)|    
|2018.03|[Tensor Core] NVIDIA Tensor Core Programmability, Performance & Precision |[[arxiv][pdf]](https://arxiv.org/pdf/1803.04014.pdf)|-|   
|2018.05|[Online Softmax] Online normalizer calculation for softmax |[[arxiv][pdf]](https://arxiv.org/pdf/1805.02867.pdf)|-|    
|2023.09|[PagedAttention] Efficient Memory Management for Large Language  Model Serving with PagedAttention |[[arxiv][pdf]](https://arxiv.org/pdf/2309.06180.pdf)|[[GitHub][vllm]](https://github.com/vllm-project/vllm) ![](https://img.shields.io/github/stars/vllm-project/vllm.svg?style=social)|   
|2023.08|[Tensor Cores] Reducing shared memory footprint to leverage high  throughput on Tensor Cores and its flexible API extension library |[[arxiv][pdf]](https://arxiv.org/pdf/2308.15152.pdf)|[[GitHub][wmma_extension]](https://github.com/wmmae/wmma_extension) ![](https://img.shields.io/github/stars/wmmae/wmma_extension.svg?style=social)|   
|2021.04|[RoPE] ROFORMER: ENHANCED TRANSFORMER WITH ROTARY  POSITION EMBEDDING |[[arxiv][pdf]](https://arxiv.org/pdf/2104.09864.pdf)|[[GitHub][transformers]](https://huggingface.co/docs/transformers/model_doc/roformer) ![](https://img.shields.io/github/stars/huggingface/transformers.svg?style=social)|   
|2022.11|[SmoothQuant] Accurate and Efficient  Post-Training Quantization for Large Language Models |[[arxiv][pdf]](https://arxiv.org/pdf/2211.10438.pdf)|[[GitHub][smoothquant]](https://github.com/mit-han-lab/smoothquant) ![](https://img.shields.io/github/stars/mit-han-lab/smoothquant.svg?style=social)|   
|2023.05|[SpecInfer] Accelerating Generative Large Language Model Serving with  Speculative Inference and Token Tree Verification |[[arxiv][pdf]](https://arxiv.org/pdf/2305.09781.pdf)|[[GitHub][FlexFlow]](https://github.com/flexflow/FlexFlow/tree/inference) ![](https://img.shields.io/github/stars/flexflow/FlexFlow.svg?style=social)|   
|2022.11|[WINT8/4] Who Says Elephants Can’t Run: Bringing Large Scale MoE Models into Cloud Scale Production |[[arxiv][pdf]](https://arxiv.org/pdf/2211.10017.pdf)|[[GitHub][FasterTransformer]](https://github.com/NVIDIA/FasterTransformer) ![](https://img.shields.io/github/stars/NVIDIA/FasterTransformer.svg?style=social)|   
|2022.06|[ZeroQuant] Efficient and Affordable Post-Training Quantization for Large-Scale Transformers |[[arxiv][pdf]](https://arxiv.org/pdf/2206.01861.pdf)|[[GitHub][DeepSpeed]](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social)|   
|2023.03|[ZeroQuant-V2] Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation|[[arxiv][pdf]](https://arxiv.org/pdf/2303.08302.pdf)|[[GitHub][DeepSpeed]](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social)|
|2023.07|[ZeroQuant-FP] A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats|[[arxiv][pdf]](https://arxiv.org/pdf/2307.09782.pdf)|[[GitHub][DeepSpeed]](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social)|


## ©️License  

GNU General Public License v3.0
