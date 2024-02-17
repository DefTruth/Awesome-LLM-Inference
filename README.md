
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
Awesome-LLM-Inference: A curated list of [📙Awesome LLM Inference Papers with Codes](#paperlist), please check [📖Contents](#paperlist) for more details. This repo is still updated frequently ~ 👨‍💻‍

## ©️Citations 

```BibTeX
@misc{Awesome-LLM-Inference@2023,
  title={Awesome-LLM-Inference: A curated list of Awesome LLM Inference Papers with codes},
  url={https://github.com/DefTruth/Awesome-LLM-Inference},
  note={Open-source software available at https://github.com/DefTruth/Awesome-LLM-Inference},
  author={Yanjun Qiu},
  year={2023}
}
```

## 🎉Download PDFs  

[@Awesome-LLM-Inference-v0.3.pdf](https://github.com/DefTruth/Awesome-LLM-Inference/releases/download/v0.3/Awesome-LLM-Inference-v0.3.pdf.zip): 500 pages, FastServe, FlashAttention 1/2, FlexGen, FP8, LLM.int8(), PagedAttention, RoPE, SmoothQuant, WINT8/4, Continuous Batching, ZeroQuant 1/2/FP, AWQ etc. 

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
* [IO/FLOPs-Aware/Sparse Attention](#IO-FLOPs-Aware-Attention-Sparse) 
* [KV Cache Scheduling/Quantize/Dropping](#KV-Cache-Scheduling-Quantize-Dropping)
* [Early-Exit/Intermediate Layer Decoding](#Early-Exit)
* [Parallel Decoding/Sampling](#Parallel-Decoding-Sampling)
* [Structured Prune/KD/Weight Sparse](#Structured_Pruning_KD_Weight_Sparse)
* [Mixture-of-Experts(MoE) LLM Inference](#Mixture_of_Experts_LLM_Inference)
* [CPU/Single GPU/Mobile Inference](#CPU-Single-GPU-Inference)
* [Non Transformer Architecture](#Non-Transformer-Architecture)
* [GEMM/Tensor Cores/WMMA/Parallel](#GEMM-Tensor-Cores-WMMA)  
* [Position Embed/Others](#Others)


### 📖LLM Algorithmic/Eval Survey ([©️back👆🏻](#paperlist))  
<div id="LLM-Algorithmic-Eval-Survey"></div>  

|Date|Title|Paper|Code|Recom|
|:---:|:---:|:---:|:---:|:---:|   
|2023.10|[Evaluating] Evaluating Large Language Models: A Comprehensive Survey(@tju.edu.cn)| [[pdf]](https://arxiv.org/pdf/2310.19736.pdf)|[[Awesome-LLMs-Evaluation]](https://github.com/tjunlp-lab/Awesome-LLMs-Evaluation-Papers)  ![](https://img.shields.io/github/stars/tjunlp-lab/Awesome-LLMs-Evaluation-Papers.svg?style=social) |⭐️ |   
|2023.11|🔥[**Runtime Performance**] Dissecting the Runtime Performance of the Training, Fine-tuning, and Inference of Large Language Models(@hkust-gz.edu.cn) | [[pdf]](https://arxiv.org/pdf/2311.03687.pdf)|⚠️|⭐️⭐️ | 
|2023.11|[ChatGPT Anniversary] ChatGPT’s One-year Anniversary: Are Open-Source Large Language Models Catching up?(@e.ntu.edu.sg)| [[pdf]](https://arxiv.org/pdf/2311.16989.pdf)|⚠️|⭐️ | 
|2023.12|[Algorithmic Survey] The Efficiency Spectrum of Large Language Models: An Algorithmic Survey(@Microsoft) | [[pdf]](https://arxiv.org/pdf/2312.00678.pdf)|⚠️|⭐️ | 
|2023.12|[Security and Privacy] A Survey on Large Language Model (LLM) Security and Privacy: The Good, the Bad, and the Ugly(@Drexel University)| [[pdf]](https://arxiv.org/pdf/2312.02003.pdf)|⚠️|⭐️ | 
|2023.12|🔥[**LLMCompass**] A Hardware Evaluation Framework for Large Language Model Inference(@princeton.edu) | [[pdf]](https://arxiv.org/pdf/2312.03134.pdf)|⚠️|⭐️⭐️ | 
|2023.12|🔥[**Efficient LLMs**] Efficient Large Language Models: A Survey(@Ohio State University etc) | [[pdf]](https://arxiv.org/pdf/2312.03863.pdf)|[[Efficient-LLMs-Survey]](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)  ![](https://img.shields.io/github/stars/AIoT-MLSys-Lab/Efficient-LLMs-Survey.svg?style=social) |⭐️⭐️ | 
|2023.12|[**Serving Survey**] Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems(@Carnegie Mellon University) | [[pdf]](https://arxiv.org/pdf/2312.15234.pdf)|⚠️|⭐️⭐️ | 
|2024.01|[Understanding LLMs] Understanding LLMs: A Comprehensive Overview from Training to Inference(@Shaanxi Normal University etc)| [[pdf]](https://arxiv.org/pdf/2401.02038.pdf) | ⚠️|⭐️⭐️ | 

### 📖LLM Train/Inference Framework ([©️back👆🏻](#paperlist))  
<div id="LLM-Train-Inference-Framework"></div>  

|Date|Title|Paper|Code|Recom|
|:---:|:---:|:---:|:---:|:---:| 
|2020.05|🔥[**Megatron-LM**] Training Multi-Billion Parameter Language Models Using Model Parallelism(@NVIDIA)|[[pdf]](https://arxiv.org/pdf/1909.08053.pdf)|[[Megatron-LM]](https://github.com/NVIDIA/Megatron-LM) ![](https://img.shields.io/github/stars/NVIDIA/Megatron-LM.svg?style=social)|⭐️⭐️ |    
|2023.03|[FlexGen] High-Throughput Generative Inference of Large Language Models  with a Single GPU(@Stanford University etc) |[[pdf]](https://arxiv.org/pdf/2303.06865.pdf)|[[FlexGen]](https://github.com/FMInference/FlexGen) ![](https://img.shields.io/github/stars/FMInference/FlexGen.svg?style=social)|⭐️ |          
|2023.05|[SpecInfer] Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification(@Peking University etc) |[[pdf]](https://arxiv.org/pdf/2305.09781.pdf)|[[FlexFlow]](https://github.com/flexflow/FlexFlow/tree/inference) ![](https://img.shields.io/github/stars/flexflow/FlexFlow.svg?style=social)|⭐️ |     
|2023.05|[FastServe] Fast Distributed Inference Serving for Large Language Models(@Peking University etc) |[[pdf]](https://arxiv.org/pdf/2305.05920.pdf)|⚠️|⭐️ |         
|2023.09|🔥[**vLLM**] Efficient Memory Management for Large Language Model Serving with PagedAttention(@UC Berkeley etc) |[[pdf]](https://arxiv.org/pdf/2309.06180.pdf)|[[vllm]](https://github.com/vllm-project/vllm) ![](https://img.shields.io/github/stars/vllm-project/vllm.svg?style=social)|⭐️⭐️ |     
|2023.09|[StreamingLLM] EFFICIENT STREAMING LANGUAGE MODELS WITH ATTENTION SINKS(@Meta AI etc)|[[pdf]](https://arxiv.org/pdf/2309.17453.pdf)|[[streaming-llm]](https://github.com/mit-han-lab/streaming-llm) ![](https://img.shields.io/github/stars/mit-han-lab/streaming-llm.svg?style=social)|⭐️ |  
|2023.09|[Medusa] Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads(@Tianle Cai etc)|[[blog]](https://sites.google.com/view/medusa-llm)|[[Medusa]](https://github.com/FasterDecoding/Medusa) ![](https://img.shields.io/github/stars/FasterDecoding/Medusa.svg?style=social)|⭐️ |    
|2023.10|🔥[**TensorRT-LLM**] NVIDIA TensorRT LLM(@NVIDIA) |[[docs]](https://nvidia.github.io/TensorRT-LLM/)|[[TensorRT-LLM]](https://github.com/NVIDIA/TensorRT-LLM) ![](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg?style=social) |⭐️⭐️ |    
|2023.11|🔥[**DeepSpeed-FastGen 2x vLLM?**] DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference(@Microsoft)| [[pdf]](https://arxiv.org/pdf/2401.08671.pdf) | [[deepspeed-fastgen]](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social) |⭐️⭐️ |  
|2023.12|🔥[**PETALS**] Distributed Inference and Fine-tuning of Large Language Models Over The Internet(@HSE Univesity etc)|[[pdf]](https://arxiv.org/pdf/2312.08361.pdf)|[[petals]](https://github.com/bigscience-workshop/petals) ![](https://img.shields.io/github/stars/bigscience-workshop/petals.svg?style=social)|⭐️⭐️ |    
|2023.10|[LightSeq] LightSeq: Sequence Level Parallelism for Distributed Training of Long Context Transformers(@UC Berkeley etc)|[[pdf]](https://arxiv.org/pdf/2310.03294.pdf)|[[LightSeq]](https://github.com/RulinShao/LightSeq) ![](https://img.shields.io/github/stars/RulinShao/LightSeq.svg?style=social)|⭐️ |    
|2023.12|[PowerInfer] PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU(@SJTU)|[[pdf]](https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf)|[[PowerInfer]](https://github.com/SJTU-IPADS/PowerInfer) ![](https://img.shields.io/github/stars/SJTU-IPADS/PowerInfer.svg?style=social)|⭐️ |    
|2024.01|[inferflow]INFERFLOW: AN EFFICIENT AND HIGHLY CONFIGURABLE INFERENCE ENGINE FOR LARGE LANGUAGE MODELS(@Tencent AI Lab)|[[pdf]](https://arxiv.org/pdf/2401.08294.pdf) | [[inferflow]](https://github.com/inferflow/inferflow) ![](https://img.shields.io/github/stars/inferflow/inferflow.svg?style=social)|⭐️ |

### 📖Continuous/In-flight Batching  ([©️back👆🏻](#paperlist))  
<div id="Continuous-In-flight-Batching"></div>    

|Date|Title|Paper|Code|Recom|
|:---:|:---:|:---:|:---:|:---:| 
|2022.07|🔥[**Continuous Batching**] Orca: A Distributed Serving System for Transformer-Based Generative Models(@Seoul National University etc) |[[pdf]](https://www.usenix.org/system/files/osdi22-yu.pdf)|⚠️|⭐️⭐️ |       
|2023.10|🔥[**In-flight Batching**] NVIDIA TensorRT LLM Batch Manager(@NVIDIA) |[[docs]](https://nvidia.github.io/TensorRT-LLM/batch_manager.html)|[[TensorRT-LLM]](https://github.com/NVIDIA/TensorRT-LLM) ![](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg?style=social) |⭐️⭐️ |    
|2023.11|🔥[**DeepSpeed-FastGen 2x vLLM?**] DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference(@Microsoft)| [[blog]](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen) | [[deepspeed-fastgen]](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social) |⭐️⭐️ |  
|2023.11|[Splitwise] Splitwise: Efficient Generative LLM Inference Using Phase Splitting(@Microsoft etc)|[[pdf]](https://arxiv.org/pdf/2311.18677.pdf)|⚠️ |⭐️ |   
|2023.12|[SpotServe] SpotServe: Serving Generative Large Language Models on Preemptible Instances(@cmu.edu etc)|[[pdf]](https://arxiv.org/pdf/2311.15566.pdf)|[[SpotServe]](https://github.com/Hsword/SpotServe) ![](https://img.shields.io/github/stars/Hsword/SpotServe.svg?style=social)|⭐️ |    
|2023.10|[LightSeq] LightSeq: Sequence Level Parallelism for Distributed Training of Long Context Transformers(@UC Berkeley etc)|[[pdf]](https://arxiv.org/pdf/2310.03294.pdf)|[[LightSeq]](https://github.com/RulinShao/LightSeq) ![](https://img.shields.io/github/stars/RulinShao/LightSeq.svg?style=social)|⭐️ |    

### 📖Weight/Activation Quantize/Compress ([©️back👆🏻](#paperlist))  
<div id="Weight-Activation-Quantize-Compress"></div>  

|Date|Title|Paper|Code|Recom|
|:---:|:---:|:---:|:---:|:---:| 
|2022.06|🔥[**ZeroQuant**] Efficient and Affordable Post-Training Quantization for Large-Scale Transformers(@Microsoft) |[[pdf]](https://arxiv.org/pdf/2206.01861.pdf)|[[DeepSpeed]](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social)|⭐️⭐️ |     
|2022.08|[FP8-Quantization] FP8 Quantization: The Power of the Exponent(@Qualcomm AI Research) | [[pdf]](https://arxiv.org/pdf/2208.09225.pdf) | ⚠️ |⭐️ |    
|2022.08|[LLM.int8()] 8-bit Matrix Multiplication  for Transformers at Scale(@Facebook AI Research etc) |[[pdf]](https://arxiv.org/pdf/2208.07339.pdf)|[[bitsandbytes]](https://github.com/timdettmers/bitsandbytes) ![](https://img.shields.io/github/stars/timdettmers/bitsandbytes.svg?style=social)|⭐️ |    
|2022.10|🔥[**GPTQ**] GPTQ: ACCURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PRE-TRAINED TRANSFORMERS(@IST Austria etc) |[[pdf]](https://arxiv.org/pdf/2210.17323.pdf) |[[gptq]](https://github.com/IST-DASLab/gptq) ![](https://img.shields.io/github/stars/IST-DASLab/gptq.svg?style=social)|⭐️⭐️ |   
|2022.11|🔥[**WINT8/4**] Who Says Elephants Can’t Run: Bringing Large Scale MoE Models into Cloud Scale Production(@NVIDIA&Microsoft) |[[pdf]](https://arxiv.org/pdf/2211.10017.pdf)|[[FasterTransformer]](https://github.com/NVIDIA/FasterTransformer) ![](https://img.shields.io/github/stars/NVIDIA/FasterTransformer.svg?style=social)|⭐️⭐️ |     
|2022.11|🔥[**SmoothQuant**] Accurate and Efficient Post-Training Quantization for Large Language Models(@MIT etc) |[[pdf]](https://arxiv.org/pdf/2211.10438.pdf)|[[smoothquant]](https://github.com/mit-han-lab/smoothquant) ![](https://img.shields.io/github/stars/mit-han-lab/smoothquant.svg?style=social)|⭐️⭐️ |     
|2023.03|[ZeroQuant-V2] Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation(@Microsoft)|[[pdf]](https://arxiv.org/pdf/2303.08302.pdf)|[[DeepSpeed]](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social)|⭐️ |  
|2023.06|🔥[**AWQ**] AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration(@MIT etc)|[[pdf]](https://browse.arxiv.org/pdf/2306.00978.pdf)|[[llm-awq]](https://github.com/mit-han-lab/llm-awq) ![](https://img.shields.io/github/stars/mit-han-lab/llm-awq.svg?style=social)|⭐️⭐️ |  
|2023.06|[SpQR] SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression(@University of Washington etc)|[[pdf]](https://browse.arxiv.org/pdf/2306.03078.pdf)|[[SpQR]](https://github.com/Vahe1994/SpQR) ![](https://img.shields.io/github/stars/Vahe1994/SpQR.svg?style=social)|⭐️ |    
|2023.06|[SqueezeLLM] SQUEEZELLM: DENSE-AND-SPARSE QUANTIZATION(@berkeley.edu) | [[pdf]](https://arxiv.org/pdf/2306.07629.pdf) | [[SqueezeLLM]](https://github.com/SqueezeAILab/SqueezeLLM) ![](https://img.shields.io/github/stars/SqueezeAILab/SqueezeLLM.svg?style=social) |⭐️ |  
|2023.07|[ZeroQuant-FP] A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats(@Microsoft)|[[pdf]](https://arxiv.org/pdf/2307.09782.pdf)|[[DeepSpeed]](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social)|⭐️ |  
|2023.09|[KV Cache FP8 + WINT4] Exploration on LLM inference performance optimization(@HPC4AI) | [[blog]](https://zhuanlan.zhihu.com/p/653735572)|⚠️|⭐️ |    
|2023.10|[FP8-LM] FP8-LM: Training FP8 Large Language Models(@Microsoft etc)| [[pdf]](https://arxiv.org/pdf/2310.18313.pdf)| [[MS-AMP]](https://github.com/Azure/MS-AMP) ![](https://img.shields.io/github/stars/Azure/MS-AMP.svg?style=social) |⭐️ |   
|2023.10|[LLM-Shearing] SHEARED LLAMA: ACCELERATING LANGUAGE MODEL PRE-TRAINING VIA STRUCTURED PRUNING(@cs.princeton.edu etc)| [[pdf]](https://arxiv.org/pdf/2310.06694.pdf) | [[LLM-Shearing]](https://github.com/princeton-nlp/LLM-Shearing) ![](https://img.shields.io/github/stars/princeton-nlp/LLM-Shearing.svg?style=social)  |⭐️ |   
|2023.10|[LLM-FP4] LLM-FP4: 4-Bit Floating-Point Quantized Transformers(@ust.hk&meta etc) | [[pdf]](https://arxiv.org/pdf/2310.16836.pdf) | [[LLM-FP4]](https://github.com/nbasyl/LLM-FP4) ![](https://img.shields.io/github/stars/nbasyl/LLM-FP4.svg?style=social) |⭐️ |    
|2023.11|[2-bit LLM] Enabling Fast 2-bit LLM on GPUs: Memory Alignment, Sparse Outlier, and Asynchronous Dequantization(@Shanghai Jiao Tong University etc) |[[pdf]](https://arxiv.org/pdf/2311.16442.pdf)|⚠️ |⭐️ | 
|2023.12|[**SmoothQuant+**] SmoothQuant+: Accurate and Efficient 4-bit Post-Training Weight Quantization for LLM(@ZTE Corporation)  | [[pdf]](https://arxiv.org/pdf/2312.03788.pdf) | [[smoothquantplus]](https://github.com/Adlik/smoothquantplus) ![](https://img.shields.io/github/stars/Adlik/smoothquantplus.svg?style=social) |⭐️ |    
|2023.11|[OdysseyLLM W4A8] A Speed Odyssey for Deployable Quantization of LLMs(@meituan.com)|[[pdf]](https://arxiv.org/pdf/2311.09550.pdf)|⚠️|⭐️ | 
|2023.12|🔥[**SparQ**] SPARQ ATTENTION: BANDWIDTH-EFFICIENT LLM INFERENCE(@graphcore.ai)|[[pdf]](https://arxiv.org/pdf/2312.04985.pdf)|⚠️|⭐️⭐️ | 
|2023.12|[Agile-Quant] Agile-Quant: Activation-Guided Quantization for Faster Inference of LLMs on the Edge(@Northeastern University&Oracle)|[[pdf]](https://arxiv.org/pdf/2312.05693.pdf)|⚠️|⭐️ | 
|2023.12|[CBQ] CBQ: Cross-Block Quantization for Large Language Models(@ustc.edu.cn)|[[pdf]](https://arxiv.org/pdf/2312.07950.pdf)|⚠️|⭐️ | 
|2023.10|[QLLM] QLLM: ACCURATE AND EFFICIENT LOW-BITWIDTH QUANTIZATION FOR LARGE LANGUAGE MODELS(@ZIP Lab&SenseTime Research etc)|[[pdf]](https://arxiv.org/pdf/2310.08041.pdf)|⚠️|⭐️ | 
|2024.01|[FP6-LLM] FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design(@Microsoft etc)|[[pdf]](https://arxiv.org/pdf/2401.14112.pdf)|⚠️|⭐️ | 

### 📖IO/FLOPs-Aware/Sparse Attention ([©️back👆🏻](#paperlist))  
<div id="IO-FLOPs-Aware-Attention-Sparse"></div>   

|Date|Title|Paper|Code|Recom|
|:---:|:---:|:---:|:---:|:---:|   
|2018.05| [Online Softmax] Online normalizer calculation for softmax(@NVIDIA) |[[pdf]](https://arxiv.org/pdf/1805.02867.pdf)|⚠️|⭐️ |    
|2019.11|🔥[MQA] Fast Transformer Decoding: One Write-Head is All You Need(@Google) | [[pdf]](https://arxiv.org/pdf/1911.02150.pdf)|⚠️|⭐️⭐️ |  
|2020.10|[Hash Attention] REFORMER: THE EFFICIENT TRANSFORMER(@Google)| [[pdf]](https://arxiv.org/pdf/2001.04451.pdf)|[[reformer]](https://github.com/google/trax/tree/master/trax/models/reformer) ![](https://img.shields.io/github/stars/google/trax.svg?style=social)|⭐️⭐️ |
|2022.05|🔥[**FlashAttention**] Fast and Memory-Efficient Exact Attention with IO-Awareness(@Stanford University etc) |[[pdf]](https://arxiv.org/pdf/2205.14135.pdf)|[[flash-attention]](https://github.com/Dao-AILab/flash-attention) ![](https://img.shields.io/github/stars/Dao-AILab/flash-attention.svg?style=social)|⭐️⭐️ |       
|2022.10|[Online Softmax] SELF-ATTENTION DOES NOT NEED O(n^2) MEMORY(@Google)| [[pdf]](https://arxiv.org/pdf/2112.05682.pdf) | ⚠️ |⭐️ |  
|2023.05|[FlashAttention] From Online Softmax to FlashAttention(@cs.washington.edu)|[[pdf]](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)|⚠️|⭐️⭐️ |   
|2023.05|[FLOP, I/O] Dissecting Batching Effects in GPT Inference(@Lequn Chen) | [[blog]](https://le.qun.ch/en/blog/2023/05/13/transformer-batching/) | ⚠️ |⭐️ |   
|2023.05|🔥🔥[**GQA**] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints(@Google) | [[pdf]](https://arxiv.org/pdf/2305.13245.pdf)|[[flaxformer]](https://github.com/google/flaxformer) ![](https://img.shields.io/github/stars/google/flaxformer.svg?style=social) |⭐️⭐️ |  
|2023.06|[Sparse FlashAttention] Faster Causal Attention Over Large Sequences Through Sparse Flash Attention(@EPFL etc) |[[pdf]](https://arxiv.org/pdf/2306.01160.pdf) | [[dynamic-sparse-flash-attention]](https://github.com/epfml/dynamic-sparse-flash-attention) ![](https://img.shields.io/github/stars/epfml/dynamic-sparse-flash-attention.svg?style=social)|⭐️ |  
|2023.07|🔥[**FlashAttention-2**] Faster Attention with Better Parallelism and Work Partitioning(@Stanford University etc) |[[pdf]](https://arxiv.org/pdf/2307.08691.pdf)|[[flash-attention]](https://github.com/Dao-AILab/flash-attention) ![](https://img.shields.io/github/stars/Dao-AILab/flash-attention.svg?style=social)|⭐️⭐️ |      
|2023.10|🔥[**Flash-Decoding**] Flash-Decoding for long-context inference(@Stanford University etc)|[[blog]](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)|[[flash-attention]](https://github.com/Dao-AILab/flash-attention) ![](https://img.shields.io/github/stars/Dao-AILab/flash-attention.svg?style=social)|⭐️⭐️ |      
|2023.11|[Flash-Decoding++] FLASHDECODING++: FASTER LARGE LANGUAGE MODEL INFERENCE ON GPUS(@Tsinghua University&Infinigence-AI) | [[pdf]](https://arxiv.org/pdf/2311.01282.pdf) | ⚠️ |⭐️ |    
|2023.01|[SparseGPT] SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot(@ISTA etc)| [[pdf]](https://arxiv.org/pdf/2301.00774.pdf)| [[sparsegpt]](https://github.com/IST-DASLab/sparsegpt) ![](https://img.shields.io/github/stars/IST-DASLab/sparsegpt.svg?style=social) |⭐️ |    
|2023.11|🔥[**HyperAttention**] HyperAttention: Long-context Attention in Near-Linear Time(@yale&Google)|[[pdf]](https://arxiv.org/pdf/2310.05869.pdf)|[hyper-attn](https://github.com/insuhan/hyper-attn)  ![](https://img.shields.io/github/stars/insuhan/hyper-attn.svg?style=social)|⭐️⭐️ |    
|2023.11|[Streaming Attention Approximation] One Pass Streaming Algorithm for Super Long Token Attention Approximation in Sublinear Space(@Adobe Research etc)|[[pdf]](https://arxiv.org/pdf/2311.14652.pdf)|⚠️ |⭐️ |  
|2023.12|🔥[**GLA**] Gated Linear Attention Transformers with Hardware-Efficient Training(@MIT-IBM Watson AI)|[[pdf]](https://arxiv.org/pdf/2312.06635.pdf)|[gated_linear_attention](https://github.com/berlino/gated_linear_attention)  ![](https://img.shields.io/github/stars/berlino/gated_linear_attention.svg?style=social)|⭐️⭐️ | 
|2023.12|[SCCA] SCCA: Shifted Cross Chunk Attention for long contextual semantic expansion(@Beihang University)| [[pdf]](https://arxiv.org/pdf/2312.07305.pdf) | ⚠️ |⭐️ |  
|2023.05|[Landmark Attention] Random-Access Infinite Context Length for Transformers(@epfl.ch)|[[pdf]](https://arxiv.org/pdf/2305.16300.pdf)|[landmark-attention](https://github.com/epfml/landmark-attention/)  ![](https://img.shields.io/github/stars/epfml/landmark-attention.svg?style=social)|⭐️⭐️ | 
|2023.12|🔥[**FlashLLM**] LLM in a flash: Efficient Large Language Model Inference with Limited Memory(@Apple)| [[pdf]](https://arxiv.org/pdf/2312.11514.pdf) | ⚠️ |⭐️⭐️ |  

### 📖KV Cache Scheduling/Quantize/Dropping ([©️back👆🏻](#paperlist))    
<div id="KV-Cache-Scheduling-Quantize-Dropping"></div>  

|Date|Title|Paper|Code|Recom|
|:---:|:---:|:---:|:---:|:---:|  
|2019.11|🔥[MQA] Fast Transformer Decoding: One Write-Head is All You Need(@Google) | [[pdf]](https://arxiv.org/pdf/1911.02150.pdf)|⚠️|⭐️⭐️ |  
|2022.06|[LTP] Learned Token Pruning for Transformers(@UC Berkeley etc)| [[pdf]](https://arxiv.org/pdf/2107.00910.pdf)|[[LTP]](https://github.com/kssteven418/LTP) ![](https://img.shields.io/github/stars/kssteven418/LTP.svg?style=social)|⭐️ |  
|2023.05|🔥🔥[**GQA**] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints(@Google) | [[pdf]](https://arxiv.org/pdf/2305.13245.pdf)|[[flaxformer]](https://github.com/google/flaxformer) ![](https://img.shields.io/github/stars/google/flaxformer.svg?style=social) |⭐️⭐️ |  
|2023.05|[KV Cache Compress] Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time(@)|[[pdf]](https://arxiv.org/pdf/2305.17118.pdf)|⚠️|⭐️⭐️ |  
|2023.06|[H2O] H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models(@Rice University etc)|[[pdf]](https://arxiv.org/pdf/2306.14048.pdf)|[[H2O]](https://github.com/FMInference/H2O) ![](https://img.shields.io/github/stars/FMInference/H2O.svg?style=social) |⭐️ |  
|2023.06|[QK-Sparse/Dropping Attention] Faster Causal Attention Over Large Sequences Through Sparse Flash Attention(@EPFL etc) |[[pdf]](https://arxiv.org/pdf/2306.01160.pdf) | [[dynamic-sparse-flash-attention]](https://github.com/epfml/dynamic-sparse-flash-attention) ![](https://img.shields.io/github/stars/epfml/dynamic-sparse-flash-attention.svg?style=social)|⭐️ |  
|2023.09|🔥🔥[**PagedAttention**] Efficient Memory Management for Large Language  Model Serving with PagedAttention(@UC Berkeley etc) |[[pdf]](https://arxiv.org/pdf/2309.06180.pdf)|[[vllm]](https://github.com/vllm-project/vllm) ![](https://img.shields.io/github/stars/vllm-project/vllm.svg?style=social)|⭐️⭐️ |     
|2023.09|[KV Cache FP8 + WINT4] Exploration on LLM inference performance optimization(@HPC4AI) | [[blog]](https://zhuanlan.zhihu.com/p/653735572)|⚠️|⭐️ |    
|2023.10|🔥[**TensorRT-LLM KV Cache FP8**] NVIDIA TensorRT LLM(@NVIDIA) |[[docs]](https://nvidia.github.io/TensorRT-LLM/precision.html)|[[TensorRT-LLM]](https://github.com/NVIDIA/TensorRT-LLM) ![](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg?style=social) |⭐️⭐️ |    
|2023.10|🔥[**Adaptive KV Cache Compress**] MODEL TELLS YOU WHAT TO DISCARD: ADAPTIVE KV CACHE COMPRESSION FOR LLMS(@illinois.edu&microsoft)|[[pdf]](https://arxiv.org/pdf/2310.01801.pdf)|⚠️|⭐️⭐️ |  
|2023.10|[CacheGen] CacheGen: Fast Context Loading for Language Model Applications(@Chicago University&Microsoft)|[[pdf]](https://arxiv.org/pdf/2310.07240.pdf)|⚠️|⭐️ |  
|2023.12|[KV-Cache Optimizations] Leveraging Speculative Sampling and KV-Cache Optimizations Together for Generative AI using OpenVINO(@Haim Barad etc) | [[pdf]](https://arxiv.org/pdf/2311.04951.pdf)|⚠️|⭐️ |    
|2023.11|🔥[Prompt Cache] PROMPT CACHE: MODULAR ATTENTION REUSE FOR LOW-LATENCY INFERENCE(@Yale University etc)|[[pdf]](https://arxiv.org/pdf/2311.04934.pdf)|⚠️|⭐️⭐️ |
|2024.01|🔥🔥[**DistKV-LLM**] Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache(@Alibaba etc)|[[pdf]](https://arxiv.org/pdf/2401.02669.pdf)|⚠️|⭐️⭐️ |
|2024.01|🔥🔥[KVQuant] KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization(@UC Berkeley)|[[pdf]](https://browse.arxiv.org/pdf/2401.18079.pdf)|[[KVQuant]](https://github.com/SqueezeAILab/KVQuant/) ![](https://img.shields.io/github/stars/SqueezeAILab/KVQuant.svg?style=social) |⭐️⭐️ |  
|2024.02|🔥🔥[Prompt Caching] Efficient Prompt Caching via Embedding Similarity(@UC Berkeley)|[[pdf]](https://arxiv.org/pdf/2402.01173.pdf)|⚠️|⭐️⭐️ |


### 📖Early-Exit/Intermediate Layer Decoding ([©️back👆🏻](#paperlist))  
<div id="Early-Exit"></div>  

|Date|Title|Paper|Code|Recom|
|:---:|:---:|:---:|:---:|:---:| 
|2020.04|[DeeBERT] DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference(@uwaterloo.ca)|[[pdf]](https://arxiv.org/pdf/2004.12993.pdf)|⚠️|⭐️ | 
|2021.06|[BERxiT] BERxiT: Early Exiting for BERT with Better Fine-Tuning and Extension to Regression(@uwaterloo.ca)|[[pdf]](https://aclanthology.org/2021.eacl-main.8.pdf)|[[berxit]](https://github.com/castorini/berxit) ![](https://img.shields.io/github/stars/castorini/berxit.svg?style=social)|⭐️ | 
|2023.10|🔥[**LITE**] Accelerating LLaMA Inference by Enabling Intermediate Layer Decoding via Instruction Tuning with LITE(@Arizona State University) | [[pdf]](https://arxiv.org/pdf/2310.18581v2.pdf)|⚠️|⭐️⭐️ | 
|2023.12|🔥🔥[**EE-LLM**] EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language Models with 3D Parallelism(@alibaba-inc.com) | [[pdf]](https://arxiv.org/pdf/2312.04916.pdf)| [[EE-LLM]](https://github.com/pan-x-c/EE-LLM) ![](https://img.shields.io/github/stars/pan-x-c/EE-LLM.svg?style=social) |⭐️⭐️ |    
|2023.10|🔥[**FREE**] Fast and Robust Early-Exiting Framework for Autoregressive Language Models with Synchronized Parallel Decoding(@KAIST AI&AWS AI)|[[pdf]](https://arxiv.org/pdf/2310.05424.pdf)| [[fast_robust_early_exit]](https://github.com/raymin0223/fast_robust_early_exit) ![](https://img.shields.io/github/stars/raymin0223/fast_robust_early_exit.svg?style=social) |⭐️⭐️ |    

### 📖Parallel Decoding/Sampling ([©️back👆🏻](#paperlist))     
<div id="Parallel-Decoding-Sampling"></div>    

|Date|Title|Paper|Code|Recom|
|:---:|:---:|:---:|:---:|:---:|   
|2018.11|🔥[**Parallel Decoding**] Blockwise Parallel Decoding for Deep Autoregressive Models(@Berkeley&Google)| [[pdf]](https://arxiv.org/pdf/1811.03115.pdf)|⚠️ |⭐️⭐️ | 
|2023.02|🔥[**Speculative Sampling**] Accelerating Large Language Model Decoding with Speculative Sampling(@DeepMind)|[[pdf]](https://arxiv.org/pdf/2302.01318.pdf)| ⚠️ |⭐️⭐️ | 
|2023.05|🔥[**Speculative Sampling**] Fast Inference from Transformers via Speculative Decoding(@Google Research etc) | [[pdf]](https://arxiv.org/pdf/2211.17192.pdf)| [[LLMSpeculativeSampling]](https://github.com/feifeibear/LLMSpeculativeSampling) ![](https://img.shields.io/github/stars/feifeibear/LLMSpeculativeSampling.svg?style=social) |⭐️⭐️ |
|2023.09|🔥[**Medusa**] Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads(@Tianle Cai etc)|[[pdf]](https://arxiv.org/pdf/2401.10774.pdf)|[[Medusa]](https://github.com/FasterDecoding/Medusa) ![](https://img.shields.io/github/stars/FasterDecoding/Medusa.svg?style=social)|⭐️⭐️ |    
|2023.10|[**OSD**] Online Speculative Decoding(@UC Berkeley etc) | [[pdf]](https://arxiv.org/pdf/2310.07177.pdf)| ⚠️ |⭐️⭐️|  
|2023.12|[**Cascade Speculative**] Cascade Speculative Drafting for Even Faster LLM Inference(@illinois.edu) | [[pdf]](https://arxiv.org/pdf/2312.11462.pdf)| ⚠️ |⭐️|   
|2024.02|🔥[LookaheadDecoding] Break the Sequential Dependency of LLM Inference Using LOOKAHEAD DECODING(@UCSD&Google&UC Berkeley)|[[pdf]](https://arxiv.org/pdf/2402.02057.pdf)| [[LookaheadDecoding]](https://github.com/hao-ai-lab/LookaheadDecoding) ![](https://img.shields.io/github/stars/hao-ai-lab/LookaheadDecoding.svg?style=social) |⭐️⭐️ |   
|2024.02| [speculative Decoding] Decoding Speculative Decoding(@cs.wisc.edu)|[[pdf]](https://arxiv.org/pdf/2402.01528.pdf)| ⚠️ |⭐️|   

### 📖Structured Prune/KD/Weight Sparse ([©️back👆🏻](#paperlist))    
<div id="Structured_Pruning_KD_Weight_Sparse"></div>    

|Date|Title|Paper|Code|Recom|
|:---:|:---:|:---:|:---:|:---:|     
|2023.12|[**FLAP**] Fluctuation-based Adaptive Structured Pruning for Large Language Models(@Chinese Academy of Sciences etc)| [[pdf]](https://arxiv.org/pdf/2312.11983.pdf)| [[FLAP]](https://github.com/CASIA-IVA-Lab/FLAP) ![](https://img.shields.io/github/stars/CASIA-IVA-Lab/FLAP.svg?style=social)|⭐️⭐️ |    
|2023.12|🔥[**LASER**] The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction(@mit.edu)|[[pdf]](https://arxiv.org/pdf/2312.13558.pdf)| [[laser]](https://github.com/pratyushasharma/laser) ![](https://img.shields.io/github/stars/pratyushasharma/laser.svg?style=social)|⭐️⭐️ |    
|2023.12|[PowerInfer] PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU(@SJTU)|[[pdf]](https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf)|[[PowerInfer]](https://github.com/SJTU-IPADS/PowerInfer) ![](https://img.shields.io/github/stars/SJTU-IPADS/PowerInfer.svg?style=social)|⭐️ |    
|2024.01|[**Admm Pruning**] Fast and Optimal Weight Update for Pruned Large Language Models(@fmph.uniba.sk)|[[pdf]](https://arxiv.org/pdf/2401.02938.pdf)|[[admm-pruning]](https://github.com/fmfi-compbio/admm-pruning) ![](https://img.shields.io/github/stars/fmfi-compbio/admm-pruning.svg?style=social)|⭐️ |    
|2024.01|[FFSplit] FFSplit: Split Feed-Forward Network For Optimizing Accuracy-Efficiency Trade-off in Language Model Inference(@1Rice University etc) | [[pdf]](https://arxiv.org/pdf/2401.04044.pdf) |  ⚠️ |⭐️|   

### 📖Mixture-of-Experts(MoE) LLM Inferencen ([©️back👆🏻](#paperlist))    
<div id="Mixture_of_Experts_LLM_Inference"></div>    

|Date|Title|Paper|Code|Recom|
|:---:|:---:|:---:|:---:|:---:|     
|2022.11|🔥[**WINT8/4**] Who Says Elephants Can’t Run: Bringing Large Scale MoE Models into Cloud Scale Production(@NVIDIA&Microsoft) |[[pdf]](https://arxiv.org/pdf/2211.10017.pdf)|[[FasterTransformer]](https://github.com/NVIDIA/FasterTransformer) ![](https://img.shields.io/github/stars/NVIDIA/FasterTransformer.svg?style=social)|⭐️⭐️ |     
|2023.12|🔥 [**Mixtral Offloading**] Fast Inference of Mixture-of-Experts Language Models with Offloading(@Moscow Institute of Physics and Technology etc)| [[pdf]](https://arxiv.org/pdf/2312.17238.pdf)| [[mixtral-offloading]](https://github.com/dvmazur/mixtral-offloading) ![](https://img.shields.io/github/stars/dvmazur/mixtral-offloading.svg?style=social)|⭐️⭐️ |
|2024.01| [MoE-Mamba] MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts(@uw.edu.pl) |  [[pdf]](https://arxiv.org/pdf/2401.04081.pdf)| ⚠️ |⭐️|   
  
### 📖CPU/Single GPU/Mobile Inference ([©️back👆🏻](#paperlist))  
<div id="CPU-Single-GPU-Inference"></div>  

|Date|Title|Paper|Code|Recom|
|:---:|:---:|:---:|:---:|:---:| 
|2023.03|[FlexGen] High-Throughput Generative Inference of Large Language Models  with a Single GPU(@Stanford University etc) |[[pdf]](https://arxiv.org/pdf/2303.06865.pdf)|[[FlexGen]](https://github.com/FMInference/FlexGen) ![](https://img.shields.io/github/stars/FMInference/FlexGen.svg?style=social)|⭐️ |          
|2023.11|[LLM CPU Inference] Efficient LLM Inference on CPUs(@intel)|[[pdf]](https://arxiv.org/pdf/2311.00502.pdf)| [[intel-extension-for-transformers]](https://github.com/intel/intel-extension-for-transformers) ![](https://img.shields.io/github/stars/intel/intel-extension-for-transformers.svg?style=social) |⭐️ |     
|2023.12|[LinguaLinked] LinguaLinked: A Distributed Large Language Model Inference System for Mobile Devices(@University of California Irvine)|[[pdf]](https://arxiv.org/pdf/2312.00388.pdf)|⚠️ |⭐️ | 
|2023.12|[OpenVINO] Leveraging Speculative Sampling and KV-Cache Optimizations Together for Generative AI using OpenVINO(@Haim Barad etc) | [[pdf]](https://arxiv.org/pdf/2311.04951.pdf)|⚠️|⭐️ | 

### 📖Non Transformer Architecture ([©️back👆🏻](#paperlist))    
<div id="Non-Transformer-Architecture"></div>  

|Date|Title|Paper|Code|Recom|
|:---:|:---:|:---:|:---:|:---:| 
|2023.05|🔥🔥[**RWKV**] RWKV: Reinventing RNNs for the Transformer Era(@Bo Peng etc) |[[pdf]](https://arxiv.org/pdf/2305.13048.pdf)|[[RWKV-LM]](https://github.com/BlinkDL/RWKV-LM) ![](https://img.shields.io/github/stars/BlinkDL/RWKV-LM.svg?style=social)|⭐️⭐️ |          
|2023.12|🔥🔥[**Mamba**] Mamba: Linear-Time Sequence Modeling with Selective State Spaces(@cs.cmu.edu etc) |[[pdf]](https://arxiv.org/pdf/2312.00752.pdf)|[[mamba]](https://github.com/state-spaces/mamba) ![](https://img.shields.io/github/stars/state-spaces/mamba.svg?style=social)|⭐️⭐️ |          

### 📖GEMM/Tensor Cores/WMMA/Parallel ([©️back👆🏻](#paperlist))    
<div id="GEMM-Tensor-Cores-WMMA"></div>  

|Date|Title|Paper|Code|Recom|
|:---:|:---:|:---:|:---:|:---:|   
|2018.03|[Tensor Core] NVIDIA Tensor Core Programmability, Performance & Precision(@KTH Royal etc) |[[pdf]](https://arxiv.org/pdf/1803.04014.pdf)|⚠️|⭐️ |
|2022.09|[FP8] FP8 FORMATS FOR DEEP LEARNING(@NVIDIA) |[[pdf]](https://arxiv.org/pdf/2209.05433.pdf)|⚠️|⭐️ |       
|2023.08|[Tensor Cores] Reducing shared memory footprint to leverage high  throughput on Tensor Cores and its flexible API extension library(@Tokyo Institute etc) |[[pdf]](https://arxiv.org/pdf/2308.15152.pdf)|[[wmma_extension]](https://github.com/wmmae/wmma_extension) ![](https://img.shields.io/github/stars/wmmae/wmma_extension.svg?style=social)|⭐️ |   
|2024.02|[QUICK] QUICK: Quantization-aware Interleaving and Conflict-free Kernel for efficient LLM inference(@SqueezeBits Inc)|[[pdf]](https://arxiv.org/pdf/2402.10076.pdf)|[[QUICK]](https://github.com/SqueezeBits/QUICK) ![](https://img.shields.io/github/stars/SqueezeBits/QUICK.svg?style=social)|⭐️⭐️ |
|2024.02|[Tensor Parallel] TP-AWARE DEQUANTIZATION(@IBM T.J. Watson Research Center)|[[pdf]](https://arxiv.org/pdf/2402.04925.pdf)|⚠️|⭐️ | 

### 📖Position Embed/Others ([©️back👆🏻](#paperlist))  
<div id="Others"></div>  

|Date|Title|Paper|Code|Recom|
|:---:|:---:|:---:|:---:|:---:|   
|2021.04|🔥[RoPE] ROFORMER: ENHANCED TRANSFORMER WITH ROTARY  POSITION EMBEDDING(@Zhuiyi Technology Co., Ltd.) |[[pdf]](https://arxiv.org/pdf/2104.09864.pdf)|[[transformers]](https://huggingface.co/docs/transformers/model_doc/roformer) ![](https://img.shields.io/github/stars/huggingface/transformers.svg?style=social)|⭐️ |     
|2022.10|[ByteTransformer] A High-Performance Transformer Boosted for Variable-Length Inputs(@ByteDance&NVIDIA)|[[pdf]](https://arxiv.org/pdf/2210.03052.pdf)|[[ByteTransformer]](https://github.com/bytedance/ByteTransformer) ![](https://img.shields.io/github/stars/bytedance/ByteTransformer.svg?style=social)|⭐️ |       


## ©️License  

GNU General Public License v3.0  

## 🎉Contribute  

Welcome to submit a PR to this repo! 

