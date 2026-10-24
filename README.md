# Self-Modelling LLM (Creative Machines Lab)

**Status:** work in progress. All results below are preliminary and come from single-seed runs.

## Research question

When one neural network learns several tasks one after another, does part of it stay the same across every task switch? We call such a persistent subnetwork a "self": a group of neurons that keep working together while the rest of the network changes to fit each new task. This repository tests that question on a small translation Transformer trained continually across three languages, and compares it with control runs that never switch tasks.

The analysis follows the method used in the Creative Machines Lab robot work, which this repository refers to as "the robot paper": correlation groups of neurons, Hungarian matching between checkpoints, and a per-neuron change score.

## Setup

**Model.** An encoder-decoder Transformer trained from scratch (`common_v5.py`): 4 encoder and 4 decoder layers, `d_model` 640, 10 heads, MLP width 2,560 with GELU, QK-norm, pre-LayerNorm, shared input and output embeddings, dropout 0.1, label smoothing 0.1.

**Tokenizer.** A SentencePiece model with 32,000 pieces (`tokenizer_v4/spm.model`). It is trained once and locked, so every run and every checkpoint uses the same vocabulary. The target language is chosen with a tag token (`<2hi>`, `<2zh>`, `<2ar>`) at the start of the decoder input.

**Tasks.** English to Hindi, English to Chinese and English to Arabic. Each vocabulary piece is assigned to a script (Devanagari, Han, Arabic, Latin or common). During training and decoding the output is restricted to the target script plus common pieces. Blocked pieces get probability zero and no gradient, so training one language does not directly push down the other languages' tokens. Hindi data comes from IITB English-Hindi (`tokenizer_train.ipynb`) and Arabic from OPUS-100 (`arabic.ipynb`). The training notebooks expect `data/en-{hi,zh,ar}/{train,val}.jsonl` to already be present.

**Phases.** Training is split into phases. Each phase trains one language with a fresh AdamW optimizer (weight decay 0.1 on weight matrices only) and this schedule:

1. 500 warmup steps to a learning rate of 2e-4.
2. A constant learning rate until the masked validation loss plateaus (less than 0.5% improvement between consecutive 4-evaluation windows, checked every 500 steps after a minimum of 2,000 steps).
3. A 2,000-step linear cooldown to 2e-5.

If validation loss goes above 1.15 times the phase best twice in a row, the phase restarts from its starting weights at half the learning rate, at most twice. One checkpoint is saved per phase.

**Runs.**
- Continual (`v4b_continual_s0`): random init (phase 0), then hi, zh, ar, hi, zh, ar, and so on.
- Control (`v4b_only_hi_s0`): random init, then Hindi in every phase.

Both runs use the same seed, so phase 1 is the same Hindi-from-scratch training in both, and the runs only diverge from phase 2. The v5 runs keep the v4b run names because they continue the v4b checkpoints.

**Evaluation.** After every phase, all three languages are evaluated, so forgetting is visible:
- chrF with greedy decoding on the first 200 sentences of FLORES+ devtest.
- Validation loss on 2,000 held-out pairs per language.

Both are reported unrestricted and script-restricted ("masked"); the tables below use the masked versions. The analysis uses the separate FLORES+ dev split as its fixed probe inputs.

## Analysis pipeline (v5)

**Pass 1: `analysis_v5_compute.py`** (GPU, threshold-free). For each checkpoint of a run:

1. **Activations.** Activations are recorded at 20 probe points (`probe_points` in `common_v5.py`): the 2,560 GELU neurons of each of the 8 MLP layers, plus the attention value vectors at the 4 encoder self-attention, 4 decoder self-attention and 4 decoder cross-attention points. The inputs are the same for every checkpoint: the English FLORES+ dev sentences for the encoder side, and teacher-forced FLORES+ dev references in all three languages for the decoder side. Up to 24,000 token positions are sampled per trace. On the decoder side each language's mean is removed per neuron before z-scoring.
2. **Correlation.** A neuron-by-neuron correlation matrix R is computed for each probe point.
3. **Spanning tree.** The maximum spanning tree of |R| is stored. Linking neurons with |r| >= tau and taking connected groups gives exactly the same groups as keeping the tree edges with weight >= tau. So any threshold tau can be analysed later on CPU without recomputing R. The largest group is called the self and the remaining alive neurons are task neurons.
4. **Health statistics.** Per-neuron std, mean and fraction of positive activations, the effective rank of R, and weight norms.
5. **Matching across checkpoints.** For each pair of consecutive checkpoints, neurons are matched with the Hungarian algorithm on activation similarity. Each matched neuron then gets:
   - an activation similarity (correlation of its traces);
   - a connectivity similarity (cosine similarity of its rows of R);
   - a change score, `change = 100 x (1 - mean of the two)`. The mean of the two similarities is the neuron's persistence.

Output: one `analysis/v5/<run>/pXXX.npz` per checkpoint, plus `weights.jsonl`, `phases.jsonl` and `log.jsonl`.

**Pass 2: `analysis_v5_interpret.py`** (GPU, run after pass 1 has finished for both runs). It works on the 8 MLP layers at chosen phases (default 18, 45, 60, 72, 84).

- **Threshold.** tau per layer is tau50: the smallest tau at which the median self covers at most 50% of the layer, pooled over phases 43 to 60 of both runs. The values used are in `analysis/v5/interpret/tau50_used.json` (0.18 to 0.23).
- **Groups.** The self at that tau, and the core: neurons that were in the self before the last switch and are still in it after.
- **Lesion test.** Each group is replaced by its mean activation, and the rise in masked validation loss is measured per language. The control is the same test on size-matched random task neurons, averaged over 3 draws.
- **Meaning stability.** For each neuron, the overlap (Jaccard) of its top-50 activating positions between one chosen phase and the next, compared for core, self and task neurons.
- **Top tokens.** The FLORES tokens that 15 core neurons and 15 random task neurons fire on most.
- **Figures.** Paper-style figures: a reordered co-activation map with per-neuron persistence, Hindi-only and continual side by side.

Output: `analysis/v5/interpret/`.

## Version history

**v3** (`common.py`, `train_continual_gpu0.ipynb`, `train_controls_gpu1.ipynb`). A smaller model (3+3 layers, `d_model` 512, MLP 2,048) with a 24,000-piece tokenizer (`tokenizer/`). The languages were Hindi, Yoruba and Chinese. All runs started from a pretrained English-Hindi checkpoint rather than random weights. The continual run cycled hi, yo, zh for 24 phases, with Hindi-only and Yoruba-only controls. The analysis used MLP neurons only and a fixed tau of 0.5 (`analysis/v3/`). At that tau the self was 1% to 10% of a layer.

**v4** (`common_v4.py`, `analysis_v4.ipynb`). Arabic replaced Yoruba, so every language has its own script and the script masks separate them fully. The model grew to the current size, QK-norm was added, and a new locked 32,000-piece tokenizer was trained (`tokenizer_v4/`). Both runs started from random weights and were planned for 18 phases, against a Hindi-only control. A z-loss on the output logits was used. The note in `common_v4.py` records that an unrestricted z-loss pushed down other languages' tokens and caused heavy forgetting. The analysis swept tau and tracked whether the self keeps the same members (`analysis/v4/`, 9 transitions).

**v4b** (`v4b_continual_gpu0.ipynb`, `v4b_control_hi_gpu1.ipynb`, `analysis_v4b.ipynb`). The same as v4 with the z-loss turned off. Both runs trained for 18 phases (6 cycles). The analysis added a table of mean persistence by where a neuron came from and where it went (self or task before and after a switch), plus per-layer co-activation figures (`analysis/v4b/`, 17 transitions).

**v5** (`common_v5.py`, `v5_continual_gpu0.ipynb`, `v5_control_hi_gpu1.ipynb`, `analysis_v5_*.py`). The same model and training recipe as v4b. Both v4b runs were resumed from phase 18 and extended towards phase 90 (30 cycles). Other changes:
- The best checkpoint per language is kept in `runs/<run>/best/`, and every checkpoint is backed up to Hugging Face in the background.
- Attention value vectors became hookable as probe points.
- The analysis was split into the threshold-free compute pass (spanning trees, so tau is chosen offline) and the interpretation pass (lesions, meaning stability, top tokens).

## Results so far

All numbers come from `summaries.txt` and the CSVs under `analysis/`. chrF and validation loss are script-restricted.

**Training (v5 runs, `summaries.txt`).**
- **Run lengths.** The Hindi-only control reached phase 90. The continual run reached phase 86.
- **Control.** Hindi chrF stays between 47.7 and 50.1 over all 90 phases, and Hindi validation loss between 1.300 and 1.356.
- **Continual, first cycle.** Hindi reaches 48.9 chrF in phase 1. The first Chinese phase reaches 24.1 and the first Arabic phase 51.9.
- **Continual, middle cycles.** Through the middle cycles, the language being trained reaches about 50 to 52 chrF (hi), 25 to 27.6 (zh) and 52 to 54 (ar). The best values are hi 51.9 (phase 52), zh 27.6 (phase 38) and ar 53.8 (phase 24).
- **Forgetting.** It is large and repeats every cycle. After the first Chinese phase Hindi falls to 8.5 chrF, and after later Chinese phases to about 20 to 26. After an Arabic phase, Hindi falls to between 0.2 and 11.
- **Continual, late phases.** From about phase 66 the trained-language scores decline. By phases 84 to 86 they are hi 39.8, zh 17.2 and ar 40.9. Hindi validation loss after a Hindi phase rises from 1.237 (phase 52) to 2.397 (phase 85).

**v4b analysis at tau 0.3** (`analysis/v4b/summary_all_taus.csv`, means over the 8 MLP layers and 17 transitions):

| run | self size (fraction of layer) | change: self | change: task | change: random | self more stable | self member overlap |
|---|---|---|---|---|---|---|
| continual | 0.111 | 39.2 | 49.0 | 48.1 | 97.1% of transitions | 0.318 |
| only_hi | 0.168 | 6.8 | 8.6 | 8.3 | 100% of transitions | 0.695 |

In both runs the self changes less than task neurons and less than size-matched random groups. In the continual run all three groups change much more than in the control. The self also keeps fewer of the same members from one phase to the next.

**v5 interpretation** (`analysis/v5/interpret/`):
- **Lesion test.** Mean-ablating the core raised own-language validation loss by 0.073 on average in the continual run, against 0.040 for size-matched random task neurons. In the Hindi-only run the figures were 0.263 and 0.088.
- **Lesion coverage.** The test is skipped when a group is larger than the pool of task neurons. Because of that, and because the interpretation pass covers only continual phases 18 to 72, the continual lesion averages rest on only a few rows.
- **Meaning stability.** Mean top-50 Jaccard was 0.247 (core), 0.243 (self) and 0.188 (task) in the continual run, and 0.407, 0.406 and 0.386 in the Hindi-only run.

These results are not yet conclusive. They come from one seed per run, and the self is defined at a chosen threshold.

## How to run

Checkpoints, FLORES+ and the training data are not in the repository (`runs/` and `data/` are ignored). The notebooks set `REPO = "/workspace/shreshth"` and pin a GPU in their first cell. Update `REPO` to your checkout path before running them.

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. **Training.** Run one notebook per GPU. Each resumes from the last saved phase, so it can be rerun at any time.
   - `v5_continual_gpu0.ipynb` on GPU 0: the continual run.
   - `v5_control_hi_gpu1.ipynb` on GPU 1: the Hindi-only control.

   Both notebooks download FLORES+ (gated, needs a Hugging Face login) and the earlier checkpoints from the Hugging Face backup if they are missing. The v4b and v3 notebooks (`v4b_*`, `train_*`) reproduce the earlier versions.
3. **Analysis pass 1.** Run one run per GPU, for example in tmux. Rerunning skips checkpoints that are already done.
   ```bash
   python -u analysis_v5_compute.py --run v4b_continual_s0 --gpu 0
   python -u analysis_v5_compute.py --run v4b_only_hi_s0  --gpu 1
   ```
4. **Analysis pass 2.** Run it after pass 1 has finished for both runs:
   ```bash
   python -u analysis_v5_interpret.py --gpu 0
   ```
   Use `--phases` to choose different phases (default `18,45,60,72,84`).

`analysis_v5_compute.ipynb` and `analysis_v5_interpret.ipynb` are notebook versions of the two scripts.

## Repository layout

```
common.py, common_v4.py, common_v5.py   shared model, tokenizer, masks, data, evaluation and training engine (v3, v4/v4b, v5)
train_continual_gpu0.ipynb              v3 continual run
train_controls_gpu1.ipynb               v3 Hindi-only and Yoruba-only controls
v4b_continual_gpu0.ipynb                v4b continual run
v4b_control_hi_gpu1.ipynb               v4b Hindi-only control
v5_continual_gpu0.ipynb                 v5 continual run (continues v4b)
v5_control_hi_gpu1.ipynb                v5 Hindi-only control (continues v4b)
analysis_v4.ipynb, analysis_v4b.ipynb   v4 and v4b analysis notebooks
analysis_v5_compute.py / .ipynb         v5 analysis, pass 1
analysis_v5_interpret.py / .ipynb       v5 analysis, pass 2
summaries.txt                           per-phase chrF and validation loss for both v5 runs
encoder_decoder_hindi.ipynb             first English-Hindi model and sanity checks
tokenizer_train.ipynb                   v3 data preparation and tokenizer
arabic_english_hindi_chinese_tokenizer_train.ipynb   v4 tokenizer
arabic.ipynb                            English-Arabic data preparation
tokenizer/, tokenizer_v4/               SentencePiece models (v3, v4 onwards)
analysis/v3, v4, v4b/                   CSV summaries and figures per version
analysis/v5/<run>/                      per-checkpoint npz files and logs from pass 1
analysis/v5/interpret/                  lesion, meaning stability, top tokens, tau50 and paper-style figures
archive/                                scratch and duplicate notebooks kept for reference
```
