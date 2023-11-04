
<div align='center'>
  <img width="500" height="250" alt="v02" src="https://github.com/DefTruth/LLMs-Inference-Papers/assets/31974251/bb136842-8054-4599-8bfe-36c36f0e997f">
</div>  

<div align='center'>
  <img src=https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg >
  <img src=https://img.shields.io/github/downloads/DefTruth/Awesome-LLM-Inference/total?color=ccf&label=downloads&logo=github&logoColor=lightgrey >
  <img src=https://img.shields.io/github/stars/DefTruth/Awesome-LLM-Inference.svg?style=social >
 </div>   

## 📒Introduction
Awesome-LLM-Inference: A small collection for Awesome LLM inference **[Papers|Blogs|Tech Report|Docs]** with codes, please check [📙Awesome LLM Inference Papers with Codes](#paperlist) for more details.

## 🎉Download PDFs  
- [LLMs-Inference-Papers-v0.1.pdf](https://github.com/DefTruth/Awesome-LLM-Inference/releases/download/v0.1/LLMs-Inference-Papers-v0.1.zip): Introduction to LLMs and LLMs inference tech, 600 pages PDF, contains Transformer, BN, LN, MQA, FlashAttention 1/2, GLM, GPT, LLaMA 1/2, LoRA, QLoRA, P-Tuning V1/V2, RoPE, SmoothQuant, WINT8/4, Continuous Batching, FP8 etc.
- [LLMs-Inference-Papers-v0.2.pdf](https://github.com/DefTruth/Awesome-LLM-Inference/releases/download/v0.2/LLMs-Inference-Papers-v0.2.zip): LLMs inference papers only, 286 pages PDF, contains ByteTransformer, FastServe, FlashAttention 1/2, FlexGen, FP8, LLM.int8(), Tensor Cores, PagedAttention, RoPE, SmoothQuant, SpecInfer, WINT8/4, Continuous Batching, ZeroQuant etc.

## 📙Awesome LLM Inference Papers with Codes

<div id="paperlist"></div>  

|Date|Title|Paper|Code|
|:---:|:---:|:---:|:---:|  
|2020.05|[Megatron-LM] Training Multi-Billion Parameter Language Models Using Model Parallelism|[[arxiv][pdf]](https://arxiv.org/pdf/1909.08053.pdf)|[[GitHub][Megatron-LM]](https://github.com/NVIDIA/Megatron-LM) ![](https://img.shields.io/github/stars/NVIDIA/Megatron-LM.svg?style=social)|   
|2022.10|[ByteTransformer] A High-Performance Transformer Boosted for Variable-Length Inputs|[[arxiv][pdf]](https://arxiv.org/pdf/2210.03052.pdf)|[[GitHub][ByteTransformer]](https://github.com/bytedance/ByteTransformer) ![](https://img.shields.io/github/stars/bytedance/ByteTransformer.svg?style=social)|   
|2022.07|[Continuous Batching] Orca: A Distributed Serving System for Transformer-Based Generative Models |[[osdi22-yu][pdf]](https://www.usenix.org/system/files/osdi22-yu.pdf)|👍|          
|2023.05|[FastServe] Fast Distributed Inference Serving for  Large Language Models |[[arxiv][pdf]](https://arxiv.org/pdf/2305.05920.pdf)|👍|       
|2022.05|[FlashAttention] Fast and Memory-Efficient Exact Attention with IO-Awareness |[[arxiv][pdf]](https://arxiv.org/pdf/2205.14135.pdf)|[[GitHub][flash-attention]](https://github.com/Dao-AILab/flash-attention) ![](https://img.shields.io/github/stars/Dao-AILab/flash-attention.svg?style=social)|     
|2023.07|[FlashAttention-2] Faster Attention with Better Parallelism and Work Partitioning |[[arxiv][pdf]](https://arxiv.org/pdf/2307.08691.pdf)|[[GitHub][flash-attention]](https://github.com/Dao-AILab/flash-attention) ![](https://img.shields.io/github/stars/Dao-AILab/flash-attention.svg?style=social)|    
|2023.03|[FlexGen] High-Throughput Generative Inference of Large Language Models  with a Single GPU |[[arxiv][pdf]](https://arxiv.org/pdf/2303.06865.pdf)|[[GitHub][FlexGen]](https://github.com/FMInference/FlexGen) ![](https://img.shields.io/github/stars/FMInference/FlexGen.svg?style=social)|       
|2022.09|[FP8] FP8 FORMATS FOR DEEP LEARNING |[[arxiv][pdf]](https://arxiv.org/pdf/2209.05433.pdf)|👍|   
|2022.08|[LLM.int8()] 8-bit Matrix Multiplication  for Transformers at Scale |[[arxiv][pdf]](https://arxiv.org/pdf/2208.07339.pdf)|[[GitHub][bitsandbytes]](https://github.com/timdettmers/bitsandbytes) ![](https://img.shields.io/github/stars/timdettmers/bitsandbytes.svg?style=social)|    
|2018.03|[Tensor Core] NVIDIA Tensor Core Programmability, Performance & Precision |[[arxiv][pdf]](https://arxiv.org/pdf/1803.04014.pdf)|👍|   
|2018.05|[Online Softmax] Online normalizer calculation for softmax |[[arxiv][pdf]](https://arxiv.org/pdf/1805.02867.pdf)|👍|    
|2023.09|[PagedAttention] Efficient Memory Management for Large Language  Model Serving with PagedAttention |[[arxiv][pdf]](https://arxiv.org/pdf/2309.06180.pdf)|[[GitHub][vllm]](https://github.com/vllm-project/vllm) ![](https://img.shields.io/github/stars/vllm-project/vllm.svg?style=social)|   
|2023.08|[Tensor Cores] Reducing shared memory footprint to leverage high  throughput on Tensor Cores and its flexible API extension library |[[arxiv][pdf]](https://arxiv.org/pdf/2308.15152.pdf)|[[GitHub][wmma_extension]](https://github.com/wmmae/wmma_extension) ![](https://img.shields.io/github/stars/wmmae/wmma_extension.svg?style=social)|   
|2021.04|[RoPE] ROFORMER: ENHANCED TRANSFORMER WITH ROTARY  POSITION EMBEDDING |[[arxiv][pdf]](https://arxiv.org/pdf/2104.09864.pdf)|[[GitHub][transformers]](https://huggingface.co/docs/transformers/model_doc/roformer) ![](https://img.shields.io/github/stars/huggingface/transformers.svg?style=social)|   
|2022.11|[SmoothQuant] Accurate and Efficient  Post-Training Quantization for Large Language Models |[[arxiv][pdf]](https://arxiv.org/pdf/2211.10438.pdf)|[[GitHub][smoothquant]](https://github.com/mit-han-lab/smoothquant) ![](https://img.shields.io/github/stars/mit-han-lab/smoothquant.svg?style=social)|   
|2023.05|[SpecInfer] Accelerating Generative Large Language Model Serving with  Speculative Inference and Token Tree Verification |[[arxiv][pdf]](https://arxiv.org/pdf/2305.09781.pdf)|[[GitHub][FlexFlow]](https://github.com/flexflow/FlexFlow/tree/inference) ![](https://img.shields.io/github/stars/flexflow/FlexFlow.svg?style=social)|   
|2022.11|[WINT8/4] Who Says Elephants Can’t Run: Bringing Large Scale MoE Models into Cloud Scale Production |[[arxiv][pdf]](https://arxiv.org/pdf/2211.10017.pdf)|[[GitHub][FasterTransformer]](https://github.com/NVIDIA/FasterTransformer) ![](https://img.shields.io/github/stars/NVIDIA/FasterTransformer.svg?style=social)|   
|2022.06|[ZeroQuant] Efficient and Affordable Post-Training Quantization for Large-Scale Transformers |[[arxiv][pdf]](https://arxiv.org/pdf/2206.01861.pdf)|[[GitHub][DeepSpeed]](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social)|   
|2023.03|[ZeroQuant-V2] Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation|[[arxiv][pdf]](https://arxiv.org/pdf/2303.08302.pdf)|[[GitHub][DeepSpeed]](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social)|
|2023.07|[ZeroQuant-FP] A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats|[[arxiv][pdf]](https://arxiv.org/pdf/2307.09782.pdf)|[[GitHub][DeepSpeed]](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social)|
|2023.09|[StreamingLLM] EFFICIENT STREAMING LANGUAGE MODELS WITH ATTENTION SINKS|[[arxiv][pdf]](https://arxiv.org/pdf/2309.17453.pdf)|[[GitHub][streaming-llm]](https://github.com/mit-han-lab/streaming-llm) ![](https://img.shields.io/github/stars/mit-han-lab/streaming-llm.svg?style=social)|
|2023.06|[AWQ] AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration|[[arxiv][pdf]](https://browse.arxiv.org/pdf/2306.00978.pdf)|[[GitHub][llm-awq]](https://github.com/mit-han-lab/llm-awq) ![](https://img.shields.io/github/stars/mit-han-lab/llm-awq.svg?style=social)|
|2023.06|[SpQR] SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression|[[arxiv][pdf]](https://browse.arxiv.org/pdf/2306.03078.pdf)|[[GitHub][SpQR]](https://github.com/Vahe1994/SpQR) ![](https://img.shields.io/github/stars/Vahe1994/SpQR.svg?style=social)|  
|2023.09|[Medusa] Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads|[[blog]](https://sites.google.com/view/medusa-llm)|[[GitHub][Medusa]](https://github.com/FasterDecoding/Medusa) ![](https://img.shields.io/github/stars/FasterDecoding/Medusa.svg?style=social)|  
|2023.05|[FLOP, I/O] Dissecting Batching Effects in GPT Inference | [[blog en/cn]](https://le.qun.ch/en/blog/2023/05/13/transformer-batching/) | 👍 |  
|2023.09|[KV Cache FP8 + WINT4] Exploration on LLM inference performance optimization | [[ZhiHu Tech Blog]](https://zhuanlan.zhihu.com/p/653735572)|👍|  
|2023.10|[Flash-Decoding] Flash-Decoding for long-context inference|[[tech report]](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)|[[GitHub][flash-attention]](https://github.com/Dao-AILab/flash-attention) ![](https://img.shields.io/github/stars/Dao-AILab/flash-attention.svg?style=social)|    
|2023.10|[TensorRT-LLM] NVIDIA TensorRT LLM |[[TensorRT-LLM’s Docs]](https://nvidia.github.io/TensorRT-LLM/)|[[GitHub][TensorRT-LLM]](https://github.com/NVIDIA/TensorRT-LLM) ![](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg?style=social) |  
|2023.10|[FP8-LM] FP8-LM: Training FP8 Large Language Models| [[arxiv][pdf]](https://arxiv.org/pdf/2310.18313.pdf)| [[GitHub][MS-AMP]](https://github.com/Azure/MS-AMP) ![](https://img.shields.io/github/stars/Azure/MS-AMP.svg?style=social) | 
|2023.10|[LLM-Shearing] SHEARED LLAMA: ACCELERATING LANGUAGE MODEL PRE-TRAINING VIA STRUCTURED PRUNING| [[arxiv][pdf]](https://arxiv.org/pdf/2310.06694.pdf) | [[GitHub][LLM-Shearing]](https://github.com/princeton-nlp/LLM-Shearing) ![](https://img.shields.io/github/stars/princeton-nlp/LLM-Shearing.svg?style=social)  | 



## ©️License  

GNU General Public License v3.0
