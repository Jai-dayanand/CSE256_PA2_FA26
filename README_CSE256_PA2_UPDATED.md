# CSE 256 — Programming Assignment 2 (Fall 2026)

**Transformer Blocks + Dissecting a Pretrained LLM**  
**Student:** Jai Dayanand  
**Course:** CSE 256 — Natural Language Processing, UC San Diego

## Overview

This submission implements and studies three components:

1. **Part 1 — Transformer encoder (30 points):** A Transformer encoder and two-layer feedforward speaker classifier, trained jointly to classify speech segments as Barack Obama (0), George W. Bush (1), or George H. W. Bush (2).
2. **Part 2 — Transformer decoder (20 points):** A word-level causal language model trained for next-token prediction. It reports perplexity on held-out speeches by the three politicians.
3. **Part 3 — Pretrained Pythia analysis (50 points):** An executed notebook examining `EleutherAI/pythia-160m-deduped` across training checkpoints using a logit lens and weight transplants.

Parts 1 and 2 implement attention and Transformer blocks directly in PyTorch, without `nn.Transformer`, `nn.TransformerEncoder`, `nn.TransformerDecoder`, or `nn.MultiheadAttention`. Part 3 uses Hugging Face Transformers to load pretrained checkpoints; analysis routines are implemented in the notebook.

## Project structure

The repository is expected to contain these files (other auxiliary/output files may also be present):

```text
CSE256_PA2_FA26/
├── README.md
├── PA2_code/
│   ├── main.py
│   ├── transformer.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── utilities.py
│   ├── CSE256_PA2_Part3_FINAL.ipynb
│   ├── speechesdataset/
│   │   ├── train_CLS.tsv
│   │   ├── test_CLS.tsv           # See filename note below
│   │   ├── train_LM.txt
│   │   ├── test_LM_obama.txt
│   │   ├── test_LM_wbush.txt
│   │   └── test_LM_hbush.txt
│   └── outputs/                  # Created by runs; may be omitted from Git
│       ├── part1/
│       ├── part2/
│       └── part3/
└── report/                       # Optional organization for final PDF
    └── CSE256_PA2_Complete_Report.pdf
```

**Filename note:** The uploaded `main.py` currently reads `PA2_code/speechesdataset/test_CLS.tsv`. The assignment handout refers to `test_CLS.txt`. Make sure the local file and the `main.py` path agree. If your actual dataset is named `test_CLS.txt`, change the path in `main.py` before running. Do not rename or substitute a different test set without checking its contents.

## Environment and installation

Use Python 3.12 (the saved Part 3 run reports Python 3.12.7). From the repository root:

```bash
python -m venv .venv
```

Activate it:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch numpy matplotlib nltk pandas jupyter ipykernel transformers safetensors huggingface_hub
```

For closer reproduction of the saved Part 3 notebook, its recorded environment uses **PyTorch 2.15.0+cpu** and **Transformers 5.22.2**. Those values describe the saved run; package installation may require platform-specific wheels or compatible versions. Avoid blindly pinning those versions if unavailable on your platform. The notebook uses `torch.float32` model weights.

Download the NLTK tokenizer data if needed:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

If the repository includes a validated `requirements.txt`, you may instead install it with `python -m pip install -r requirements.txt`. The command above is the explicit fallback based on the imported modules in the uploaded files.

## Run Part 1 — Speaker classification

From the repository root:

```bash
python -u PA2_code/main.py --part part1
```

Or from inside `PA2_code`:

```bash
python -u main.py --part part1
```

**Configuration from `main.py`:** seed 42; batch size 16; maximum sequence length 32; embedding dimension 64; 2 attention heads; 4 Transformer blocks; feedforward hidden size 100; Adam learning rate 0.001; **15 epochs**.

The encoder uses learned token and position embeddings, bidirectional self-attention with padding-key masking, residual connections, layer norms, and masked mean pooling. A 64 → 100 → 3 feedforward classifier is trained jointly with cross-entropy loss.

**Generated files** in `PA2_code/outputs/part1/`:

- `epoch_metrics.csv` — training loss and test accuracy after each epoch
- `accuracy_curve.png` — 15-epoch test-accuracy plot
- `sentence_1_attention.png`, `sentence_2_attention.png` — attention heatmaps
- `summary.json`, `vocabulary.json`, `classifier_final.pt` — run metadata and saved model

The code checks that attention rows sum to one across layers and heads. The reported final test accuracy in the submitted Part A report is **83.87%** after epoch 15 (best observed: **86.13%** at epoch 11). The reported encoder parameter count is **482,192** (encoder plus classifier: **488,995**).

## Run Part 2 — Causal decoder language model

From the repository root:

```bash
python -u PA2_code/main.py --part part2
```

Or from inside `PA2_code`:

```bash
python -u main.py --part part2
```

**Configuration:** seed 42; batch size 16; sequence length 32; embedding dimension 64; 2 attention heads; 4 blocks; feedforward hidden size 100 with ReLU; Adam learning rate 0.001; **500 optimizer iterations**; perplexity evaluated every **100 iterations**.

The decoder uses causal attention masking so tokens cannot attend to future positions. It trains on next-token cross-entropy and computes perplexity as `exp(mean token cross-entropy)`. The evaluation routine uses up to the **first 200 batches of sliding windows per corpus**, not a whole-file unique-token estimate. The code also checks that changing the final token does not alter predictions at earlier positions and that future-token attention probabilities are zero.

**Generated files** in `PA2_code/outputs/part2/`:

- `perplexity_metrics.csv` — training and three held-out perplexities at iterations 100, 200, 300, 400, 500
- `perplexity_curve.png` — perplexity curves
- `sentence_1_attention.png`, `sentence_2_attention.png` — causal attention heatmaps
- `summary.json`, `vocabulary.json`, `decoder_final.pt` — run metadata and saved model

**Reported results at iteration 500:**

| Corpus | Perplexity |
|---|---:|
| Training | 183.19 |
| Obama | 391.65 |
| George W. Bush | 472.84 |
| George H. W. Bush | 404.10 |

Total decoder-language-model parameters: **848,402**, including **366,210** LM-head parameters. Training processes **256,000 target positions** (overlapping windows, not necessarily unique corpus tokens).

## Run Part 3 — Pythia logit lens and transplants

The submitted Part 3 notebook is `PA2_code/CSE256_PA2_Part3_FINAL.ipynb` (rename or update the path here if your final repository uses a different filename). It should be **submitted as an executed `.ipynb` with all outputs embedded**.

From the repository root:

```bash
jupyter notebook PA2_code/CSE256_PA2_Part3_FINAL.ipynb
```

Run all cells in order to reproduce the experiments. The notebook searches for `speechesdataset/test_LM_obama.txt` relative to the notebook working directory or for `PA2_code/speechesdataset/test_LM_obama.txt` from the project root. Ensure the data file exists in one of those locations.

**Model:** `EleutherAI/pythia-160m-deduped` (12 layers, 12 heads, hidden size 768). The notebook uses the pretrained model's own tokenizer and float32 weights, not the word-level NLTK tokenizer from Parts 1–2. Checkpoints are downloaded from Hugging Face, so internet access and enough disk space are required. The experiments can be memory- and time-intensive on CPU.

- **3.1 Reading Pythia:** Two Markdown answers referencing sections of Biderman et al. (2023).
- **3.2.1 Final-checkpoint logit lens:** Layer-by-position prediction grids for the specified Obama sentence and a second original sentence; top-1 next-token accuracy over `test_LM_obama.txt` for layers 0–12; exact values printed for layers 0, 6 and 12.
- **3.2.2 Across pretraining:** Accuracy-by-layer curves for `step512`, `step1000`, `step8000`, and `step143000` plotted together, with a Markdown interpretation.
- **3.3 Layer transplants:** Replace the token embedding, each of 12 Transformer blocks, or the unembedding of the intact `step143000` model with corresponding weights from `step0`, `step1000`, `step8000`, and `step32000`. Recompute test loss for every pair and restore original weights after each transplant. Display a numerically annotated heatmap using a logarithmic color scale and explain the results.

The notebook writes artifacts to `PA2_code/outputs/part3/`, including:

```text
run_config.json
baseline.json
final_accuracy.csv
final_accuracy.png
checkpoint_accuracy.csv
checkpoint_accuracy.png
grid_obama.png
grid_obama.npz
grid_original.png
grid_original.npz
transplant_deltas.csv
transplant_losses.csv
transplant_heatmap.png
```

**Saved-run reference values:** 6,017 Pythia tokens (6,016 next-token transitions), final-checkpoint logit-lens accuracy at layer 0 = **0.016622%**, layer 6 = **4.704122%**, and layer 12 = **34.674202%**. The intact final-model baseline cross-entropy is **3.4617656152 nats/token**. The figures and CSVs in the report reflect the saved execution, not a rerun performed by this README.

## Report and Gradescope submission

The assignment requires **two deliverables**:

1. **Code submission:** All source files for Parts 1–2, a README with run instructions, the data or means to access it as permitted, and the fully executed Part 3 `.ipynb` with visible outputs. Ensure module imports and relative paths work from the submitted directory.
2. **Single report PDF:** **Part A (Parts 1–2, about five pages)** followed by **Part B (a PDF export of the same executed Part 3 notebook)**. There is no stated five-page limit on Part B. The notebook PDF must match the notebook version submitted with the code, including all numbers, figures, and explanations.

**Do not substitute the standalone Part 3 CSVs, PNGs, or a manually re-created report for the executed notebook.** These additional artifacts may be helpful for reproducibility, but the assignment grades Part 3 from the executed notebook and its corresponding export. Before submission, confirm that the `.ipynb` and Part B PDF show the same execution outputs.

## Reproducibility notes

- Both Parts 1–2 use seed 42 and compute their own word-level vocabulary from training data only.
- Training results can vary across machines, PyTorch builds, and execution environments; saved output files record the results used in the accompanying report.
- Part 3 uses the same fixed Pythia-token evaluation protocol for all checkpoints and transplants: 6,016 next-token transitions, 128 prediction positions per window, and a fresh context at each window boundary.
- Part 3 cross-entropy is measured on pretrained-model subword tokens and is not directly comparable to Part 2 word-level perplexity.
- Avoid committing `.venv/`, `.DS_Store`, cached model checkpoints, Hugging Face caches, or large intermediate `.pt` files unless explicitly requested for grading. Keep the executed notebook intact.
