# Third-party code and assets

## Vendored files

Each vendored part keeps its own license; the license texts are in the `LICENSES/` files the table names.

| File | Origin | License |
|---|---|---|
| `src/lattice/models/edm_unet.py` | NVlabs/edm, `training/networks.py` (Karras et al., 2022); only SongUNet and the layers it uses are kept, the persistence decorators removed | CC BY-NC-SA 4.0, Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES (the file keeps NVIDIA's header; `LICENSES/edm-CC-BY-NC-SA-4.0.txt`). This file, and any work derived from it, may be used for non-commercial purposes only. |
| `src/lattice/models/sit.py` | willisma/SiT, `models.py` (Ma et al., 2024); timm layers taken from `timm_layers.py` | MIT, Copyright (c) Meta Platforms, Inc. and affiliates (`LICENSES/SiT-MIT.txt`) |
| `src/lattice/models/timm_layers.py` | `PatchEmbed`, `Attention`, `Mlp` as in huggingface/pytorch-image-models 0.9, shortened to the arguments they take here | Apache-2.0, Copyright 2019 Ross Wightman (`LICENSES/timm-Apache-2.0.txt`) |
| `center_crop_arr` in `src/lattice/data/prepare_imagenet.py` | openai/guided-diffusion, `guided_diffusion/image_datasets.py` (ADM's crop) | MIT, Copyright (c) 2021 OpenAI (`LICENSES/guided-diffusion-MIT.txt`) |
| the negative prompt in `src/lattice/models/skyreels.py` | the default negative prompt of `generate_video_df.py`, SkyworkAI/SkyReels-V2, copied unchanged | Skywork Community License |

## Fetched by the setup scripts (not redistributed)

| Asset | Source | Used for |
|---|---|---|
| text8 | Matt Mahoney's text8 (the first 10^8 characters of an English Wikipedia dump) | the text8 models |
| CIFAR-10 | torchvision's download (Krizhevsky, 2009) | the pixel models and their FID reference |
| ImageNet-1k (ILSVRC2012) training set | you download it under the ImageNet terms | ImageNet-256 latents; the MAR-B kernel reads of Figure 4 (b) and Table 2 |
| `stabilityai/sd-vae-ft-ema` | Hugging Face hub | ImageNet-256 latents and decoding |
| `evaluations/evaluator.py`, `classify_image_graph_def.pb`, `VIRTUAL_imagenet256_labeled.npz` | openai/guided-diffusion and its reference batches | ImageNet-256 FID, sFID, IS, precision, recall |
| LTH14/mar at a pinned commit, the MAR-B weights, its KL-16 VAE and FID statistics | the MAR authors' repository and release | Tables 2 and 3, Figure 4 (b), Figure 5 (b) |
| `GSAI-ML/LLaDA-8B-Base` | Hugging Face hub (revision written to `REVISIONS` by the setup) | Tables 4 and 24 to 27 |
| `gpt2-large` | Hugging Face hub | generative perplexity |
| GSM8K, HumanEval, the WikiText-103 test file | Hugging Face hub and the authors' releases | task accuracy, pass@1, the LLaDA kernel read |
| SkyworkAI/SkyReels-V2 at commit 9351d131, `SkyReels-V2-DF-1.3B-540P`, VBench | the authors' repositories and weights | Table 28 |
| torch-fidelity's Inception weights | fetched by torch-fidelity on first use | CIFAR-10 FID |

Each asset keeps its own license; the setup scripts only download and verify it.
