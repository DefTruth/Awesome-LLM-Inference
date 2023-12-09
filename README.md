
<div align='center'>
  <img width="450" height="250" alt="v02" src="https://github.com/DefTruth/LLMs-Inference-Papers/assets/31974251/bb136842-8054-4599-8bfe-36c36f0e997f">  
<a href="https://star-history.com/#DefTruth/Awesome-LLM-Inference&Date">
  <picture align='center'>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=DefTruth/Awesome-LLM-Inference&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=DefTruth/Awesome-LLM-Inference&type=Date" />
    <img width="350" height="250" alt="Star History Chart" src="https://api.star-history.com/svg?repos=DefTruth/Awesome-LLM-Inference&type=Date" />
  </picture>
</a>
</div>

<div align='center'>
  <img src=https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg >
  <img src=https://img.shields.io/github/downloads/DefTruth/Awesome-LLM-Inference/total?color=ccf&label=downloads&logo=github&logoColor=lightgrey >
  <img src=https://img.shields.io/github/forks/DefTruth/Awesome-LLM-Inference.svg?style=social >
  <img src=https://img.shields.io/github/stars/DefTruth/Awesome-LLM-Inference.svg?style=social >
  <img src=https://img.shields.io/badge/PDF-avaliable-brightgreen.svg >
  <img src=https://img.shields.io/badge/release-v0.3-yellow.svg >
  <img src=https://img.shields.io/badge/License-GPLv3.0-turquoise.svg >
 </div>   

## 📒Introduction
Awesome-LLM-Inference: A small collection for [📙Awesome LLM Inference Papers with Codes](#paperlist). ❤️Star🌟👆🏻this repo to support me if it does any helps to you~

## ©️Citations 

```BibTeX
@misc{Awesome-LLM-Inference@2023,
  title={Awesome-LLM-Inference: A small collection for Awesome LLM Inference Papers with codes},
  url={https://github.com/DefTruth/Awesome-LLM-Inference},
  note={Open-source software available at https://github.com/DefTruth/Awesome-LLM-Inference},
  author={Yanjun Qiu},
  year={2023}
}
```

## 🎉Download PDFs  

[Awesome-LLM-Inference-v0.3.pdf](https://github.com/DefTruth/Awesome-LLM-Inference/releases/download/v0.3/Awesome-LLM-Inference-v0.3.pdf.zip): 500 pages, contains ByteTransformer, FastServe, FlashAttention 1/2, FlexGen, FP8, LLM.int8(), Tensor Cores, PagedAttention, RoPE, SmoothQuant, SpecInfer, WINT8/4, Continuous Batching, ZeroQuant 1/2/FP, AWQ, FlashDecoding, FlashDecoding++, FP8-LM, LLM-FP4, StreamLLM etc. 

<div align='center'>
<img src=https://github.com/DefTruth/Awesome-LLM-Inference/assets/31974251/0ed77e9d-a1eb-4095-9a82-bad624964e55 >
</div>   

## 📙Awesome LLM Inference Papers with Codes   
<div id="paperlist"></div>  

### 📖Contents 
* [LLM Algorithmic/Eval Survey](#LLM-Algorithmic-Eval-Survey)
* [LLM Train/Inference Framework](#LLM-Train-Inference-Framework)
* [Weight/Activation Quantize/Compress](#Weight-Activation-Quantize-Compress)
* [Continuous/In-flight Batching](#Continuous-In-flight-Batching)
* [IO/FLOPs-Aware Attention Optimization](#IO-FLOPs-Aware-Attention-Optimization) 
* [KV Cache Scheduling/Quantize/Compress](#KV-Cache-Scheduling-Quantize-Compress)
* [GEMM、Tensor Cores、WMMA](#GEMM-Tensor-Cores-WMMA)  
* [LLM CPU/Single GPU/Mobile Inference](#LLM-CPU-Single-GPU-Inference)
* [Non Transformer Architecture](#Non-Transformer-Architecture)
* [Sampling、Position Embed、Others](#Others)

### 📖LLM Algorithmic/Eval Survey   
<div id="LLM-Algorithmic-Eval-Survey"></div>  

|Date|Title|Paper|Code|Recommend|
|:---:|:---:|:---:|:---:|:---:|   
|2023.10|[Evaluating] Evaluating Large Language Models: A Comprehensive Survey| [[arxiv][pdf]](https://arxiv.org/pdf/2310.19736.pdf)|[[GitHub][Awesome-LLMs-Evaluation]](https://github.com/tjunlp-lab/Awesome-LLMs-Evaluation-Papers)  ![](https://img.shields.io/github/stars/tjunlp-lab/Awesome-LLMs-Evaluation-Papers.svg?style=social) |⭐️⭐️⭐️ |   
|2023.11|🔥🔥[**Runtime Performance**] Dissecting the Runtime Performance of the Training, Fine-tuning, and Inference of Large Language Models | [[arxiv][pdf]](https://arxiv.org/pdf/2311.03687.pdf)|⚠️|⭐️⭐️⭐️⭐️⭐️ | 
|2023.11|[ChatGPT Anniversary] ChatGPT’s One-year Anniversary: Are Open-Source Large Language Models Catching up?| [[arxiv][pdf]](https://arxiv.org/pdf/2311.16989.pdf)|⚠️|⭐️⭐️⭐️ | 
|2023.12|[Algorithmic Survey] The Efficiency Spectrum of Large Language Models: An Algorithmic Survey | [[arxiv][pdf]](https://arxiv.org/pdf/2312.00678.pdf)|⚠️|⭐️⭐️⭐️ | 
|2023.12|[Security and Privacy] A Survey on Large Language Model (LLM) Security and Privacy: The Good, the Bad, and the Ugly| [[arxiv][pdf]](https://arxiv.org/pdf/2312.02003.pdf)|⚠️|⭐️⭐️⭐️ | 
|2023.12|🔥🔥[**LLMCompass**] A Hardware Evaluation Framework for Large Language Model Inference | [[arxiv][pdf]](https://arxiv.org/pdf/2312.03134.pdf)|⚠️|⭐️⭐️⭐️⭐️⭐️ | 
|2023.12|🔥🔥[**Efficient LLMs**] Efficient Large Language Models: A Survey | [[arxiv][pdf]](https://arxiv.org/pdf/2312.03863.pdf)|[[GitHub][Efficient-LLMs-Survey]](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)  ![](https://img.shields.io/github/stars/AIoT-MLSys-Lab/Efficient-LLMs-Survey.svg?style=social) |⭐️⭐️⭐️⭐️⭐️ | 

### 📖LLM Train/Inference Framework  
<div id="LLM-Train-Inference-Framework"></div>  

|Date|Title|Paper|Code|Recommend|
|:---:|:---:|:---:|:---:|:---:| 
|2020.05|🔥🔥[**Megatron-LM**] Training Multi-Billion Parameter Language Models Using Model Parallelism|[[arxiv][pdf]](https://arxiv.org/pdf/1909.08053.pdf)|[[GitHub][Megatron-LM]](https://github.com/NVIDIA/Megatron-LM) ![](https://img.shields.io/github/stars/NVIDIA/Megatron-LM.svg?style=social)|⭐️⭐️⭐️⭐️⭐️ |    
|2023.03|[FlexGen] High-Throughput Generative Inference of Large Language Models  with a Single GPU |[[arxiv][pdf]](https://arxiv.org/pdf/2303.06865.pdf)|[[GitHub][FlexGen]](https://github.com/FMInference/FlexGen) ![](https://img.shields.io/github/stars/FMInference/FlexGen.svg?style=social)|⭐️⭐️⭐️ |          
|2023.05|[SpecInfer] Accelerating Generative Large Language Model Serving with  Speculative Inference and Token Tree Verification |[[arxiv][pdf]](https://arxiv.org/pdf/2305.09781.pdf)|[[GitHub][FlexFlow]](https://github.com/flexflow/FlexFlow/tree/inference) ![](https://img.shields.io/github/stars/flexflow/FlexFlow.svg?style=social)|⭐️⭐️⭐️ |     
|2023.05|[FastServe] Fast Distributed Inference Serving for  Large Language Models |[[arxiv][pdf]](https://arxiv.org/pdf/2305.05920.pdf)|⚠️|⭐️⭐️⭐️ |         
|2023.09|🔥🔥[**vLLM**] Efficient Memory Management for Large Language  Model Serving with PagedAttention |[[arxiv][pdf]](https://arxiv.org/pdf/2309.06180.pdf)|[[GitHub][vllm]](https://github.com/vllm-project/vllm) ![](https://img.shields.io/github/stars/vllm-project/vllm.svg?style=social)|⭐️⭐️⭐️⭐️⭐️ |     
|2023.09|[StreamingLLM] EFFICIENT STREAMING LANGUAGE MODELS WITH ATTENTION SINKS|[[arxiv][pdf]](https://arxiv.org/pdf/2309.17453.pdf)|[[GitHub][streaming-llm]](https://github.com/mit-han-lab/streaming-llm) ![](https://img.shields.io/github/stars/mit-han-lab/streaming-llm.svg?style=social)|⭐️⭐️⭐️ |  
|2023.09|[Medusa] Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads|[[blog]](https://sites.google.com/view/medusa-llm)|[[GitHub][Medusa]](https://github.com/FasterDecoding/Medusa) ![](https://img.shields.io/github/stars/FasterDecoding/Medusa.svg?style=social)|⭐️⭐️⭐️ |    
|2023.10|🔥🔥[**TensorRT-LLM**] NVIDIA TensorRT LLM |[[TensorRT-LLM’s Docs]](https://nvidia.github.io/TensorRT-LLM/)|[[GitHub][TensorRT-LLM]](https://github.com/NVIDIA/TensorRT-LLM) ![](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg?style=social) |⭐️⭐️⭐️⭐️⭐️ |    
|2023.11|🔥🔥[**DeepSpeed-FastGen 2x vLLM?**] DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference| [[github][blog]](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen) | [[GitHub][deepspeed-fastgen]](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social) |⭐️⭐️⭐️⭐️⭐️ |  

### 📖Continuous/In-flight Batching  
<div id="Continuous-In-flight-Batching"></div>    

|Date|Title|Paper|Code|Recommend|
|:---:|:---:|:---:|:---:|:---:| 
|2022.07|🔥🔥[**Continuous Batching**] Orca: A Distributed Serving System for Transformer-Based Generative Models |[[osdi22-yu][pdf]](https://www.usenix.org/system/files/osdi22-yu.pdf)|⚠️|⭐️⭐️⭐️⭐️⭐️ |       
|2023.10|🔥🔥[**In-flight Batching**] NVIDIA TensorRT LLM Batch Manager |[[TensorRT-LLM’s Docs]](https://nvidia.github.io/TensorRT-LLM/batch_manager.html)|[[GitHub][TensorRT-LLM]](https://github.com/NVIDIA/TensorRT-LLM) ![](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg?style=social) |⭐️⭐️⭐️⭐️⭐️ |    
|2023.11|🔥🔥[**DeepSpeed-FastGen 2x vLLM?**] DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference| [[github][blog]](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen) | [[GitHub][deepspeed-fastgen]](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social) |⭐️⭐️⭐️⭐️⭐️ |  
|2023.11|[Splitwise] Splitwise: Efficient Generative LLM Inference Using Phase Splitting|[[arxiv][pdf]](https://arxiv.org/pdf/2311.18677.pdf)|⚠️ |⭐️⭐️⭐️ |   

### 📖Weight/Activation Quantize/Compress
<div id="Weight-Activation-Quantize-Compress"></div>  

|Date|Title|Paper|Code|Recommend|
|:---:|:---:|:---:|:---:|:---:| 
|2022.06|🔥🔥[**ZeroQuant**] Efficient and Affordable Post-Training Quantization for Large-Scale Transformers |[[arxiv][pdf]](https://arxiv.org/pdf/2206.01861.pdf)|[[GitHub][DeepSpeed]](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social)|⭐️⭐️⭐️⭐️⭐️ |     
|2022.08|[FP8-Quantization] FP8 Quantization: The Power of the Exponent | [[arxiv][pdf]](https://arxiv.org/pdf/2208.09225.pdf) | ⚠️ |⭐️⭐️⭐️ |    
|2022.08|[LLM.int8()] 8-bit Matrix Multiplication  for Transformers at Scale |[[arxiv][pdf]](https://arxiv.org/pdf/2208.07339.pdf)|[[GitHub][bitsandbytes]](https://github.com/timdettmers/bitsandbytes) ![](https://img.shields.io/github/stars/timdettmers/bitsandbytes.svg?style=social)|⭐️⭐️⭐️ |    
|2022.10|🔥🔥[**GPTQ**] GPTQ: ACCURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PRE-TRAINED TRANSFORMERS |[[arxiv][pdf]](https://arxiv.org/pdf/2210.17323.pdf) |[[GitHub][gptq]](https://github.com/IST-DASLab/gptq) ![](https://img.shields.io/github/stars/IST-DASLab/gptq.svg?style=social)|⭐️⭐️⭐️⭐️⭐️ |   
|2022.11|🔥🔥[**WINT8/4**] Who Says Elephants Can’t Run: Bringing Large Scale MoE Models into Cloud Scale Production |[[arxiv][pdf]](https://arxiv.org/pdf/2211.10017.pdf)|[[GitHub][FasterTransformer]](https://github.com/NVIDIA/FasterTransformer) ![](https://img.shields.io/github/stars/NVIDIA/FasterTransformer.svg?style=social)|⭐️⭐️⭐️⭐️⭐️ |     
|2022.11|🔥🔥[**SmoothQuant**] Accurate and Efficient  Post-Training Quantization for Large Language Models |[[arxiv][pdf]](https://arxiv.org/pdf/2211.10438.pdf)|[[GitHub][smoothquant]](https://github.com/mit-han-lab/smoothquant) ![](https://img.shields.io/github/stars/mit-han-lab/smoothquant.svg?style=social)|⭐️⭐️⭐️⭐️⭐️ |     
|2023.03|[ZeroQuant-V2] Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation|[[arxiv][pdf]](https://arxiv.org/pdf/2303.08302.pdf)|[[GitHub][DeepSpeed]](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social)|⭐️⭐️⭐️ |  
|2023.06|🔥🔥[**AWQ**] AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration|[[arxiv][pdf]](https://browse.arxiv.org/pdf/2306.00978.pdf)|[[GitHub][llm-awq]](https://github.com/mit-han-lab/llm-awq) ![](https://img.shields.io/github/stars/mit-han-lab/llm-awq.svg?style=social)|⭐️⭐️⭐️⭐️⭐️ |  
|2023.06|[SpQR] SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression|[[arxiv][pdf]](https://browse.arxiv.org/pdf/2306.03078.pdf)|[[GitHub][SpQR]](https://github.com/Vahe1994/SpQR) ![](https://img.shields.io/github/stars/Vahe1994/SpQR.svg?style=social)|⭐️⭐️⭐️ |    
|2023.06|[SqueezeLLM] SQUEEZELLM: DENSE-AND-SPARSE QUANTIZATION | [[arxiv][pdf]](https://arxiv.org/pdf/2306.07629.pdf) | [[GitHub][SqueezeLLM]](https://github.com/SqueezeAILab/SqueezeLLM) ![](https://img.shields.io/github/stars/SqueezeAILab/SqueezeLLM.svg?style=social) |⭐️⭐️⭐️ |  
|2023.07|[ZeroQuant-FP] A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats|[[arxiv][pdf]](https://arxiv.org/pdf/2307.09782.pdf)|[[GitHub][DeepSpeed]](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social)|⭐️⭐️⭐️ |  
|2023.09|[KV Cache FP8 + WINT4] Exploration on LLM inference performance optimization | [[ZhiHu Tech Blog]](https://zhuanlan.zhihu.com/p/653735572)|⚠️|⭐️⭐️⭐️ |    
|2023.10|[FP8-LM] FP8-LM: Training FP8 Large Language Models| [[arxiv][pdf]](https://arxiv.org/pdf/2310.18313.pdf)| [[GitHub][MS-AMP]](https://github.com/Azure/MS-AMP) ![](https://img.shields.io/github/stars/Azure/MS-AMP.svg?style=social) |⭐️⭐️⭐️ |   
|2023.10|[LLM-Shearing] SHEARED LLAMA: ACCELERATING LANGUAGE MODEL PRE-TRAINING VIA STRUCTURED PRUNING| [[arxiv][pdf]](https://arxiv.org/pdf/2310.06694.pdf) | [[GitHub][LLM-Shearing]](https://github.com/princeton-nlp/LLM-Shearing) ![](https://img.shields.io/github/stars/princeton-nlp/LLM-Shearing.svg?style=social)  |⭐️⭐️⭐️ |   
|2023.10|[LLM-FP4] LLM-FP4: 4-Bit Floating-Point Quantized Transformers | [[arxiv][pdf]](https://arxiv.org/pdf/2310.16836.pdf) | [[GitHub][LLM-FP4]](https://github.com/nbasyl/LLM-FP4) ![](https://img.shields.io/github/stars/nbasyl/LLM-FP4.svg?style=social) |⭐️⭐️⭐️ |    
|2023.11|[2-bit LLM] Enabling Fast 2-bit LLM on GPUs: Memory Alignment, Sparse Outlier, and Asynchronous Dequantization |[[arxiv][pdf]](https://arxiv.org/pdf/2311.16442.pdf)|⚠️ |⭐️⭐️⭐️ | 
|2023.12|[**SmoothQuant+**] SmoothQuant+: Accurate and Efficient 4-bit Post-Training Weight Quantization for LLM  | [[arxiv][pdf]](https://arxiv.org/pdf/2312.03788.pdf) | [[GitHub][smoothquantplus]](https://github.com/Adlik/smoothquantplus) ![](https://img.shields.io/github/stars/Adlik/smoothquantplus.svg?style=social) |⭐️⭐️⭐️ |    

### 📖IO/FLOPs-Aware Attention Optimization
<div id="IO-FLOPs-Aware-Attention-Optimization"></div>   

|Date|Title|Paper|Code|Recommend|
|:---:|:---:|:---:|:---:|:---:|   
|2018.05| [Online Softmax] Online normalizer calculation for softmax |[[arxiv][pdf]](https://arxiv.org/pdf/1805.02867.pdf)|⚠️|⭐️⭐️⭐️ |    
|2019.11|🔥🔥[MQA] Fast Transformer Decoding: One Write-Head is All You Need | [[arxiv][pdf]](https://arxiv.org/pdf/1911.02150.pdf)|⚠️|⭐️⭐️⭐️⭐️⭐️ |  
|2022.05|🔥🔥[**FlashAttention**] Fast and Memory-Efficient Exact Attention with IO-Awareness |[[arxiv][pdf]](https://arxiv.org/pdf/2205.14135.pdf)|[[GitHub][flash-attention]](https://github.com/Dao-AILab/flash-attention) ![](https://img.shields.io/github/stars/Dao-AILab/flash-attention.svg?style=social)|⭐️⭐️⭐️⭐️⭐️ |       
|2022.10|[Online Softmax] SELF-ATTENTION DOES NOT NEED O(n^2) MEMORY| [[arxiv][pdf]](https://arxiv.org/pdf/2112.05682.pdf) | ⚠️ |⭐️⭐️⭐️ |  
|2023.05|[FlashAttention] From Online Softmax to FlashAttention|[[cse599m][flashattn.pdf]](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)|⚠️|⭐️⭐️⭐️⭐️⭐️ |   
|2023.05|[FLOP, I/O] Dissecting Batching Effects in GPT Inference | [[blog en/cn]](https://le.qun.ch/en/blog/2023/05/13/transformer-batching/) | ⚠️ |⭐️⭐️⭐️ |   
|2023.05|🔥🔥🔥[**GQA**] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | [[arxiv][pdf]](https://arxiv.org/pdf/2305.13245.pdf)|[[GitHub][flaxformer]](https://github.com/google/flaxformer) ![](https://img.shields.io/github/stars/google/flaxformer.svg?style=social) |⭐️⭐️⭐️⭐️⭐️ |  
|2023.06|[Sparse FlashAttention] Faster Causal Attention Over Large Sequences Through Sparse Flash Attention |[[arxiv][pdf]](https://arxiv.org/pdf/2306.01160.pdf) | [[GitHub][dynamic-sparse-flash-attention]](https://github.com/epfml/dynamic-sparse-flash-attention) ![](https://img.shields.io/github/stars/epfml/dynamic-sparse-flash-attention.svg?style=social)|⭐️⭐️⭐️ |  
|2023.07|🔥🔥[**FlashAttention-2**] Faster Attention with Better Parallelism and Work Partitioning |[[arxiv][pdf]](https://arxiv.org/pdf/2307.08691.pdf)|[[GitHub][flash-attention]](https://github.com/Dao-AILab/flash-attention) ![](https://img.shields.io/github/stars/Dao-AILab/flash-attention.svg?style=social)|⭐️⭐️⭐️⭐️⭐️ |      
|2023.10|🔥🔥[**Flash-Decoding**] Flash-Decoding for long-context inference|[[tech report]](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)|[[GitHub][flash-attention]](https://github.com/Dao-AILab/flash-attention) ![](https://img.shields.io/github/stars/Dao-AILab/flash-attention.svg?style=social)|⭐️⭐️⭐️⭐️⭐️ |      
|2023.11|[Flash-Decoding++] FLASHDECODING++: FASTER LARGE LANGUAGE MODEL INFERENCE ON GPUS | [[arxiv][pdf]](https://arxiv.org/pdf/2311.01282.pdf) | ⚠️ |⭐️⭐️⭐️ |    
|2023.01|[SparseGPT] SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot| [[arxiv][pdf]](https://arxiv.org/pdf/2301.00774.pdf)| [[GitHub][sparsegpt]](https://github.com/IST-DASLab/sparsegpt) ![](https://img.shields.io/github/stars/IST-DASLab/sparsegpt.svg?style=social) |⭐️⭐️⭐️ |    
|2023.11|🔥🔥[**HyperAttention**] HyperAttention: Long-context Attention in Near-Linear Time|[[arxiv][pdf]](https://arxiv.org/pdf/2310.05869.pdf)|⚠️ |⭐️⭐️⭐️⭐️⭐️ |    
|2023.11|[Streaming Attention Approximation] One Pass Streaming Algorithm for Super Long Token Attention Approximation in Sublinear Space|[[arxiv][pdf]](https://arxiv.org/pdf/2311.14652.pdf)|⚠️ |⭐️⭐️⭐️ |

### 📖KV Cache Scheduling/Quantize/Compress  
<div id="KV-Cache-Scheduling-Quantize-Compress"></div>  

|Date|Title|Paper|Code|Recommend|
|:---:|:---:|:---:|:---:|:---:|  
|2019.11|🔥🔥[MQA] Fast Transformer Decoding: One Write-Head is All You Need | [[arxiv][pdf]](https://arxiv.org/pdf/1911.02150.pdf)|⚠️|⭐️⭐️⭐️⭐️⭐️ |  
|2023.05|🔥🔥🔥[**GQA**] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | [[arxiv][pdf]](https://arxiv.org/pdf/2305.13245.pdf)|[[GitHub][flaxformer]](https://github.com/google/flaxformer) ![](https://img.shields.io/github/stars/google/flaxformer.svg?style=social) |⭐️⭐️⭐️⭐️⭐️ |  
|2023.05|[KV Cache Compress] Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time|[[arxiv][pdf]](https://arxiv.org/pdf/2305.17118.pdf)|⚠️|⭐️⭐️⭐️⭐️⭐️ |  
|2023.06|[H2O] H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models|[[arxiv][pdf]](https://arxiv.org/pdf/2306.14048.pdf)|[[GitHub][H2O]](https://github.com/FMInference/H2O) ![](https://img.shields.io/github/stars/FMInference/H2O.svg?style=social) |⭐️⭐️⭐️ |  
|2023.09|🔥🔥🔥[**PagedAttention**] Efficient Memory Management for Large Language  Model Serving with PagedAttention |[[arxiv][pdf]](https://arxiv.org/pdf/2309.06180.pdf)|[[GitHub][vllm]](https://github.com/vllm-project/vllm) ![](https://img.shields.io/github/stars/vllm-project/vllm.svg?style=social)|⭐️⭐️⭐️⭐️⭐️ |     
|2023.09|[KV Cache FP8 + WINT4] Exploration on LLM inference performance optimization | [[ZhiHu Tech Blog]](https://zhuanlan.zhihu.com/p/653735572)|⚠️|⭐️⭐️⭐️ |    
|2023.10|🔥🔥[**TensorRT-LLM KV Cache FP8**] NVIDIA TensorRT LLM |[[TensorRT-LLM’s Docs]](https://nvidia.github.io/TensorRT-LLM/precision.html)|[[GitHub][TensorRT-LLM]](https://github.com/NVIDIA/TensorRT-LLM) ![](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg?style=social) |⭐️⭐️⭐️⭐️⭐️ |    
|2023.10|🔥🔥[**Adaptive KV Cache Compress**] MODEL TELLS YOU WHAT TO DISCARD: ADAPTIVE KV CACHE COMPRESSION FOR LLMS|[[arxiv][pdf]](https://arxiv.org/pdf/2310.01801.pdf)|⚠️|⭐️⭐️⭐️⭐️⭐️ |  
|2023.10|[CacheGen] CacheGen: Fast Context Loading for Language Model Applications|[[arxiv][pdf]](https://arxiv.org/pdf/2310.07240.pdf)|⚠️|⭐️⭐️⭐️ |  
|2023.12|[KV-Cache Optimizations] Leveraging Speculative Sampling and KV-Cache Optimizations Together for Generative AI using OpenVINO | [[arxiv][pdf]](https://arxiv.org/pdf/2311.04951.pdf)|⚠️|⭐️⭐️⭐️ | 

### 📖GEMM、Tensor Cores、WMMA  
<div id="GEMM-Tensor-Cores-WMMA"></div>  

|Date|Title|Paper|Code|Recommend|
|:---:|:---:|:---:|:---:|:---:|   
|2018.03|[Tensor Core] NVIDIA Tensor Core Programmability, Performance & Precision |[[arxiv][pdf]](https://arxiv.org/pdf/1803.04014.pdf)|⚠️|⭐️⭐️⭐️ |
|2022.09|[FP8] FP8 FORMATS FOR DEEP LEARNING |[[arxiv][pdf]](https://arxiv.org/pdf/2209.05433.pdf)|⚠️|⭐️⭐️⭐️ |       
|2023.08|[Tensor Cores] Reducing shared memory footprint to leverage high  throughput on Tensor Cores and its flexible API extension library |[[arxiv][pdf]](https://arxiv.org/pdf/2308.15152.pdf)|[[GitHub][wmma_extension]](https://github.com/wmmae/wmma_extension) ![](https://img.shields.io/github/stars/wmmae/wmma_extension.svg?style=social)|⭐️⭐️⭐️ |     

### 📖LLM CPU/Single GPU/Mobile Inference
<div id="LLM-CPU-Single-GPU-Inference"></div>  

|Date|Title|Paper|Code|Recommend|
|:---:|:---:|:---:|:---:|:---:| 
|2023.03|[FlexGen] High-Throughput Generative Inference of Large Language Models  with a Single GPU |[[arxiv][pdf]](https://arxiv.org/pdf/2303.06865.pdf)|[[GitHub][FlexGen]](https://github.com/FMInference/FlexGen) ![](https://img.shields.io/github/stars/FMInference/FlexGen.svg?style=social)|⭐️⭐️⭐️ |          
|2023.11|[LLM CPU Inference] Efficient LLM Inference on CPUs|[[arxiv][pdf]](https://arxiv.org/pdf/2311.00502.pdf)| [[GitHub][intel-extension-for-transformers]](https://github.com/intel/intel-extension-for-transformers) ![](https://img.shields.io/github/stars/intel/intel-extension-for-transformers.svg?style=social) |⭐️⭐️⭐️ |     
|2023.12|[LinguaLinked] LinguaLinked: A Distributed Large Language Model Inference System for Mobile Devices|[[arxiv][pdf]](https://arxiv.org/pdf/2312.00388.pdf)|⚠️ |⭐️⭐️⭐️ | 
|2023.12|[OpenVINO] Leveraging Speculative Sampling and KV-Cache Optimizations Together for Generative AI using OpenVINO | [[arxiv][pdf]](https://arxiv.org/pdf/2311.04951.pdf)|⚠️|⭐️⭐️⭐️ | 

### 📖Non Transformer Architecture  
<div id="Non-Transformer-Architecture"></div>  

|Date|Title|Paper|Code|Recommend|
|:---:|:---:|:---:|:---:|:---:| 
|2023.05|🔥🔥🔥[**RWKV**] RWKV: Reinventing RNNs for the Transformer Era |[[arxiv][pdf]](https://arxiv.org/pdf/2305.13048.pdf)|[[GitHub][RWKV-LM]](https://github.com/BlinkDL/RWKV-LM) ![](https://img.shields.io/github/stars/BlinkDL/RWKV-LM.svg?style=social)|⭐️⭐️⭐️⭐️⭐️ |          
|2023.12|🔥🔥🔥[**Mamba**] Mamba: Linear-Time Sequence Modeling with Selective State Spaces |[[arxiv][pdf]](https://arxiv.org/pdf/2312.00752.pdf)|[[GitHub][mamba]](https://github.com/state-spaces/mamba) ![](https://img.shields.io/github/stars/state-spaces/mamba.svg?style=social)|⭐️⭐️⭐️⭐️⭐️ |          

### 📖Sampling、Position Embed、Others
<div id="Others"></div>  

|Date|Title|Paper|Code|Recommend|
|:---:|:---:|:---:|:---:|:---:|   
|2019.11|🔥🔥[MQA] Fast Transformer Decoding: One Write-Head is All You Need | [[arxiv][pdf]](https://arxiv.org/pdf/1911.02150.pdf)|⚠️|⭐️⭐️⭐️⭐️⭐️ |  
|2023.05|🔥🔥🔥[**GQA**] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | [[arxiv][pdf]](https://arxiv.org/pdf/2305.13245.pdf)|[[GitHub][flaxformer]](https://github.com/google/flaxformer) ![](https://img.shields.io/github/stars/google/flaxformer.svg?style=social) |⭐️⭐️⭐️⭐️⭐️ | 
|2021.04|🔥🔥[RoPE] ROFORMER: ENHANCED TRANSFORMER WITH ROTARY  POSITION EMBEDDING |[[arxiv][pdf]](https://arxiv.org/pdf/2104.09864.pdf)|[[GitHub][transformers]](https://huggingface.co/docs/transformers/model_doc/roformer) ![](https://img.shields.io/github/stars/huggingface/transformers.svg?style=social)|⭐️⭐️⭐️ |     
|2022.10|[ByteTransformer] A High-Performance Transformer Boosted for Variable-Length Inputs|[[arxiv][pdf]](https://arxiv.org/pdf/2210.03052.pdf)|[[GitHub][ByteTransformer]](https://github.com/bytedance/ByteTransformer) ![](https://img.shields.io/github/stars/bytedance/ByteTransformer.svg?style=social)|⭐️⭐️⭐️ |       
|2023.09|🔥🔥[StreamingLLM] EFFICIENT STREAMING LANGUAGE MODELS WITH ATTENTION SINKS|[[arxiv][pdf]](https://arxiv.org/pdf/2309.17453.pdf)|[[GitHub][streaming-llm]](https://github.com/mit-han-lab/streaming-llm) ![](https://img.shields.io/github/stars/mit-han-lab/streaming-llm.svg?style=social)|⭐️⭐️⭐️ |  
|2023.09|🔥🔥[Medusa] Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads|[[blog]](https://sites.google.com/view/medusa-llm)|[[GitHub][Medusa]](https://github.com/FasterDecoding/Medusa) ![](https://img.shields.io/github/stars/FasterDecoding/Medusa.svg?style=social)|⭐️⭐️⭐️ |    


## ©️License  

GNU General Public License v3.0  

## 🎉Contribute  

Welcome to submit a PR to this repo! 

