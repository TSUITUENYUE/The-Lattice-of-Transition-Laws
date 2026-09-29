# The Lattice of Transition Laws

Code for the paper. The corruption lattice contains diffusion, autoregression and every schedule
between them as paths through one family of transition laws. A schedule's dependence cost is the
total correlation its parallel steps discard, and it equals the schedule's divergence from exact
sampling. For data that are Markov on a graph and dependent along its paths, the fewest zero-cost
steps equal the graph's treedepth. With fewer steps, a kernel of pairwise dependence estimated from
pretrained weights predicts how schedules rank before decoding. One model trained on the whole
lattice decodes both corners and the schedules between them.

This repository trains the text8, CIFAR-10 and ImageNet-256 models of the paper, decodes released
image, language and video models under the schedules the paper compares, and estimates the kernel
from each model's weights. Every table and figure of the paper that reports a measurement has one
script in `scripts/`.

## Layout

    scripts/            one script per table or figure of the paper (tableNN.sh, figureNN.sh)
    setup/              data, released weights and evaluator preparation (prepare_*.sh)
    src/lattice/
        core/           the selection rules and the noise levels
        models/         the text8 networks, EDM's U-Net and SiT (vendored), the released models' wrappers
        data/           text8, CIFAR-10 and the ImageNet latent banks, with their preparation
        training/       the text8 recipes, EDM's recipe and the lattice training measures
        sampling/       lattice paths and the rules installed into the released samplers
        evaluation/     bits per character, FID, generative perplexity, GSM8K, HumanEval, VBench
        experiments/    every command-line entry point, and the reduction behind each display
        utils/          result files
    LICENSES/           the licence texts of the vendored code

## Setup

Python 3.11 and a CUDA GPU (the paper's runs used PyTorch 2.4.1 with CUDA 12.1).

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt     # the exact versions of the paper's runs
    pip install -e .

Every script reads its paths from the environment (defaults in brackets):

| Variable | Meaning |
|---|---|
| `DATA_ROOT` | datasets, released weights and evaluator assets [`./data`] |
| `OUTPUT_ROOT` | results, one directory per table or figure (`tableNN/`, `figureNN/`) [`./outputs`] |
| `CKPT_ROOT` | trained networks, shared between scripts [`$OUTPUT_ROOT/checkpoints`] |
| `NUM_GPUS` | GPUs to run independent jobs on at once [`1`] |
| `PYTHON` | the interpreter [`python`] |
| `SANDBOX` | where HumanEval programs run: `bwrap` [default], `unshare`, `docker` or `podman` (with `SANDBOX_IMAGE=sha256:...`) |

Paths must not contain spaces. A script that is interrupted can be run again: finished stages are
skipped, and trainings, decodes and reads resume where they stopped with their random states.

## Data and released weights

    bash setup/prepare_text8.sh                         # text8, split 90M / 5M / 5M
    bash setup/prepare_cifar10.sh                       # CIFAR-10 and its FID reference
    IMAGENET_TRAIN=/path/to/ILSVRC2012/train bash setup/prepare_imagenet.sh   # ImageNet-256 latent banks
    bash setup/prepare_adm_evaluator.sh                 # ADM's ImageNet evaluation suite
    bash setup/prepare_llada.sh                         # LLaDA-8B-Base, GPT-2-large, GSM8K, HumanEval, WikiText-103 test
    IMAGENET_TRAIN=/path/to/ILSVRC2012/train bash setup/prepare_mar.sh        # MAR at a pinned commit, MAR-B, its own environment
    bash setup/prepare_video.sh                         # SkyReels-V2 at a pinned commit, DF-1.3B-540P, VBench, its own environment

A script needs only the setup its display reads; each checks for its inputs and names the setup
script that is missing. HumanEval programs are executed only inside a sandbox with no network, a
read-only filesystem and a private temporary directory; `llada_humaneval check-sandbox` verifies
this before any candidate runs.

## Reproducing the paper

Each script trains what its display needs, runs the reads, and writes the display's rows to
`$OUTPUT_ROOT/tableNN/` or `$OUTPUT_ROOT/figureNN/`: for example `bash scripts/table05.sh` writes
`$OUTPUT_ROOT/table05/table05.csv`. The rows and columns carry the paper's labels. A table or figure
in two parts writes one file per part (`table03_text8.csv` and `table03_mar.csv` for the two halves
of Table 3), and a reading that a caption or the text quotes beside a table goes to a further file
in the same directory (`table07_budget.csv`). Sections are numbered as in the paper and appendices
lettered. Models and reads are shared: a script that finds a model or a read finished by another
script reuses it. Costs are the paper's measurements where recorded.

| Section | Display | Script | Trains | Reads |
|---|---|---|---|---|
| 1 | Figure 1 | none | a drawing of the schedules, not a measurement | |
| 2.2 | Figure 2 | `figure02.sh` | the four CIFAR-10 pixel models; the four 85M text8 models | the pixel corner cells and samples; exact chains and masked ELBOs |
| 4 | Figure 3 | none | a drawing of the rules, not a measurement | |
| 6 | Table 1 | none | the definitions of the selection rules, not a measurement | |
| 6.1 | Table 2 | `table02.sh` | nothing | the MAR-B displacement and information kernels (about 3.5 L40-hours together), their floors on the CPU, and the MAR-B cells of Table 3 |
| 6.1 | Figure 4 | `figure04.sh` | the text8 lattice model | its pair kernel; the MAR-B kernel |
| 6.2 | Table 3 | `table03.sh` | the text8 lattice model | left: decodes of every rule at 1 to 64 steps, three seeds; right: 38 MAR-B cells of FID-50K (guidance 2.9, temperature 1), five reveal orders at 8 to 256 steps, random and midpoint at 8 and 16 steps with seeds 1 and 2 as well, nested at 8, 16 and 32, spread then raster at 8 and 16; the table averages three seeds where it says so (a 256-step cell 16 to 18 L40-hours) |
| 6.2 | Figure 5 | `figure05.sh` | the text8 lattice model | its decodes; the MAR-B cells of Table 3, drawn as FID-50K over our read of MAR-B's own random order at 256 steps |
| 6.2 | Table 4 | `table04.sh` | nothing | the LLaDA kernel; 15 GSM8K cells (8 steps about half an hour, 128 steps about 9 L40-hours); generative perplexity, three seeds |
| 6.3 | Table 5 | `table05.sh` | the text8 lattice model | same-schedule chains, the kernel at five mask rates, the pairwise estimate |
| 6.5 | Table 6 | `table06.sh` | the graded and absorbing text8 models (about 64 L40-hours each) | complete decodes at equal steps |
| B.1 | Figure 6 | `figure06.sh` | the CIFAR-10 iid-levels model | intermediate states and level maps as arrays |
| B.2 | Table 7 | `table07.sh` | four CIFAR-10 pixel models, 30 to 40 L40-hours each, and one carried to 120k steps | 28 FID-50K cells, four 120k cells (10,000 and 50,000 samples), the held-out objective |
| B.2 | Table 8 | `table08.sh` | two SiT-B/2 models, 400k steps (about 25 B200-hours each) | five FID-50K cells with ADM's suite |
| B.3 | Table 9 | `table09.sh` | the causal, masked and two-stream text8 models | 16 draws of the readable points |
| C.1 | Table 10 | `table10.sh` | the text8 lattice model | the pair kernel by distance and word boundary |
| C.1 | Table 11 | `table11.sh` | the text8 lattice model | total correlation of the rules' sets |
| C.1 | Table 12 | `table12.sh` | the text8 lattice model | the kernel at five mask rates |
| C.1 | Table 13 | `table13.sh` | the 85M and the 26M text8 lattice models | the 26M model's kernel against the 85M model's costs |
| C.1 | Table 14 | `table14.sh` | the text8 lattice model | same-schedule chains |
| C.1 | Table 15 | `table15.sh` | the text8 lattice model | decodes, chains, pair and triplet kernels |
| C.2 | Table 16 | `table16.sh` | the text8 lattice model | its pair kernel with 0 to 4 of the characters between the pair revealed, five mask rates, three seeds |
| C.2 | Table 17 | `table17.sh` | the text8 lattice model | the kernel of Table 16, the chains of Table 14, the decode and chain of midpoint with ties spread, the two estimates |
| C.3 | Table 18 | `table18.sh` | the text8 lattice model | decodes at 1 to 64 steps |
| C.5 | Table 19 | `table19.sh` | the text8 lattice model | confidence decodes, their sets, the estimates |
| C.6 | Table 20 | `table20.sh` | the text8 lattice model | decodes under five width profiles |
| C.7 | Table 21 | `table21.sh` | the graded, absorbing and lattice text8 models | channel decodes and chains, channel kernels, the estimates |
| C.7 | Table 22 | `table22.sh` | the graded and absorbing text8 models | channel kernels |
| C.8 | Table 23 | `table23.sh` | the text8 lattice and causal models | free generation scored by the causal model |
| C.9 | Table 24 | `table24.sh` | nothing | the samples of Table 25 |
| C.9 | Table 25 | `table25.sh` | nothing | generative perplexity, 5 rules at 4 budgets, three seeds (about an hour per seed) |
| C.10 | Table 26 | `table26.sh` | the text8 lattice model | timed decodes of LLaDA and the text8 model (times depend on the card) |
| C.11 | Table 27 | `table27.sh` | nothing | 15 HumanEval cells, then the sandboxed scoring; GSM8K under the authors' sampling and at 256 steps |
| D | Table 28 | `table28.sh` | nothing | 15 SkyReels-V2 rollouts of 773 frames (40 to 90 minutes each on an RTX 4090, longer at greater memory depth), VBench, the frame kernel |

The text8 lattice model trains 600,000 steps (about 200 L40-hours); every read of it takes minutes.

## Protocol notes

- text8: 85M models (width 768, 12 layers, 12 heads, windows of 256 characters, batch 512, dropout
  0.2, seed 0); the causal, masked, graded and absorbing models train 200,000 steps at learning rate
  3e-4, the two-stream model 200,000 at 1e-4, the lattice model 600,000 at 1.5e-4. Each is read at
  its validation minimum (averaged weights in eval mode, fixed validation text); the 26M lattice
  model at its last step. Reads use the test split. Decodes are teacher forced on 256 windows with
  192 characters revealed and 64 decoded; the random rule averages seeds 0-2 (the chains 0-15). The
  selection sets the estimates walk are chosen on the GPU the decodes ran on. The masked ELBO uses
  512 windows and 32 draws with every row at its own masking rate.
- Selection rules, by their names in Table 1. Contiguous, random, spread, dilated and confidence
  select as the table states; on LLaDA, spread is farthest-point selection. The others:
  - Separator starts a step at the middle masked coordinate and adds, one at a time, the masked
    coordinate farthest from those already chosen, at most one per masked run until every run has
    one, then tops up to the step width.
  - Word-aware spread (Table 23) is the separator rule with words as the segments. Word boundaries
    come from the decoded spaces and from the model's own probability of a space above one half.
  - Midpoint takes the midpoint of every masked run; on LLaDA it tops up when the runs run out.
    Midpoint with ties spread (Table 17) advances the midpoints of the longest runs and spreads its
    choices among runs of equal length.
  - Min-distance is confidence with a minimum distance between the coordinates of a step: distance 8
    on text8, and 6 on LLaDA, where 6 is the kernel's decay length, the first distance at which the
    kernel falls to a tenth of its value at distance 1.
  - Separator constraint is confidence limited to one coordinate per masked run.
  - The two constrained confidence rules satisfy their constraint first and fill the step by
    confidence; every step records whether the constraint had to be relaxed.
  - On MAR-B, midpoint reveals next the token farthest from those already revealed; low-discrepancy
    is a Halton order; nested takes the middle row and column first, then each quarter; spread then
    raster applies spread to the first quarter of the grid, then rows.
- CIFAR-10 pixel lattice: 16 patches of 8 x 8 and 128 transitions; one read of 50,000 class-balanced
  samples per cell at seed 0, FID against the training set with torch-fidelity.
- ImageNet-256: SiT-B/2 on sd-vae-ft-ema latents, batch 256; 50,000 samples, ADM's suite.
- LLaDA-8B-Base: released weights; every prompt decoded on its own, without padding; greedy unless a
  temperature is given; GSM8K five-shot with examples from the full training split.
- MAR-B: released weights, guidance 2.9, temperature 1, 100 diffusion steps per token, 50 images per
  class, MAR's own FID statistics. MAR reveals a permutation from its end, so the midpoint,
  low-discrepancy, nested and spread-then-raster orders, built first-revealed-first, are reversed
  before MAR reads them.
- SkyReels-V2-DF-1.3B-540P: synchronised schedule with causal block 1, 773 frames, every frame of a
  rollout denoised before it is scored; VBench subject consistency, background consistency, motion
  smoothness and imaging quality, background consistency with CLIP in float32 and 32 frames per
  encoding call.

## Licences

The repository is MIT-licensed (`LICENSE`), except the vendored code below, which keeps its own
licence. `LICENSES/` holds the licence texts the table names, and `THIRD_PARTY_NOTICES.md` lists the
origins of the vendored code and the assets the setup scripts fetch.

| File | Origin | Licence |
|---|---|---|
| `src/lattice/models/edm_unet.py` | NVlabs/edm | CC BY-NC-SA 4.0, non-commercial use only (`LICENSES/edm-CC-BY-NC-SA-4.0.txt`) |
| `src/lattice/models/sit.py` | willisma/SiT | MIT (`LICENSES/SiT-MIT.txt`) |
| `src/lattice/models/timm_layers.py` | huggingface/pytorch-image-models | Apache-2.0 (`LICENSES/timm-Apache-2.0.txt`) |
| `center_crop_arr` in `src/lattice/data/prepare_imagenet.py` | openai/guided-diffusion | MIT (`LICENSES/guided-diffusion-MIT.txt`) |
| the negative prompt in `src/lattice/models/skyreels.py` | SkyworkAI/SkyReels-V2 | Skywork Community License |
