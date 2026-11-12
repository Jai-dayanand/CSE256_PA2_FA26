
# CSE 256 — Programming Assignment 2
## Transformer Blocks + Dissecting a Pretrained LLM

**Student:** Jai Dayanand  
**University:** University of California, San Diego  
**Course:** CSE 256 — Natural Language Processing  
**Quarter:** Fall 2026

---

## 1. Project Overview

This repository contains the implementation and experimental analysis for Programming Assignment 2 of CSE 256.

The assignment investigates Transformer architectures through three complementary tasks:

1. **Part 1 — Transformer Encoder (30 points):** Implement a Transformer encoder from scratch and train it jointly with a feedforward classifier to identify the speaker of a political speech segment.
2. **Part 2 — Transformer Decoder (20 points):** Implement a causal, GPT-style Transformer decoder from scratch and train it for word-level next-token prediction.
3. **Part 3 — Dissecting a Pretrained LLM (50 points):** Investigate the internal representations of Pythia-160M at different pretraining checkpoints using the logit lens and layer-transplant experiments.

Parts 1 and 2 use custom implementations of multihead self-attention, Transformer blocks, positional embeddings, and feedforward networks. No built-in PyTorch Transformer or multihead-attention layers are used.

Part 3 uses Hugging Face Transformers to load pretrained models, while the logit-lens and layer-transplant analysis procedures are implemented independently.

---

## 2. Repository Structure

```text
CSE256_PA2_FA26/
│
├── README.md
│
├── PA2_code/
│   ├── main.py
│   ├── transformer.py
│   ├── dataset.py
│   ├── tokenizer.py
│   ├── utilities.py
│   ├── CSE256_PA2_Part3_FINAL.ipynb
│   │
│   ├── speechesdataset/
│   │   ├── train_CLS.tsv
│   │   ├── test_CLS.tsv
│   │   ├── train_LM.txt
│   │   ├── test_LM_obama.txt
│   │   ├── test_LM_wbush.txt
│   │   └── test_LM_hbush.txt
│   │
│   └── outputs/
│       ├── part1/
│       ├── part2/
│       └── part3/
│
└── report/
    └── CSE256_PA2_Complete_Report.pdf
```

**Note:** The filenames above describe the intended final project layout. The classification code currently references `test_CLS.tsv`, while the assignment handout calls the test file `test_CLS.txt`. Verify the actual filename and ensure the path in `main.py` matches it.

The `outputs/` directories are created automatically when the corresponding experiments run.

---

## 3. Environment and Dependencies

### Python

The saved Part 3 execution used Python 3.12.7.

The recorded experimental environment includes:

| Component | Version / Configuration |
|---|---|
| Python | 3.12.7 |
| PyTorch | 2.15.0+cpu |
| Transformers | 5.22.2 |
| Device | CPU |
| Random seed | 42 |
| Pretrained model dtype | float32 |

These are the versions recorded in the saved execution; installation may require compatible platform-specific packages.

### Required Libraries

- Python
- PyTorch
- NumPy
- Matplotlib
- NLTK
- Pandas
- Transformers
- Hugging Face Hub
- Safetensors
- Jupyter Notebook
- IPython

### Installation

Clone the repository:

```bash
git clone https://github.com/Jai-dayanand/CSE256_PA2_FA26.git
cd CSE256_PA2_FA26
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch numpy matplotlib nltk pandas transformers huggingface_hub safetensors jupyter ipykernel ipython
```

If a validated `requirements.txt` is included in the repository, use:

```bash
python -m pip install -r requirements.txt
```

Install the required NLTK tokenizer resources:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

The Part 3 notebook requires internet access to retrieve Pythia checkpoints from Hugging Face.

---

# Part 1 — Transformer Encoder and Speaker Classification

## 4. Objective

The objective is to classify speech segments according to the politician who delivered them.

The classification labels are:

| Label | Speaker |
|---|---|
| 0 | Barack Obama |
| 1 | George W. Bush |
| 2 | George H. W. Bush |

The model consists of a Transformer encoder followed by a feedforward classification network.

Both components are trained jointly using supervised cross-entropy loss.

## 5. Encoder Architecture

### Input Representations

Each word is converted into a token ID using the provided word-level tokenizer.

The input representation is formed by adding a learned token embedding and a learned absolute positional embedding:

```text
Input embedding = Token embedding + Position embedding
```

The encoder uses four Transformer blocks, each containing:

1. Multihead self-attention with two attention heads
2. Scaled dot-product attention
3. Padding-key masking
4. Residual connections
5. Layer normalization
6. A two-layer feedforward network with ReLU activation

The attention computation is implemented as:

```text
Attention(Q, K, V) =
    softmax(QK^T / sqrt(d_k)) V
```

The encoder employs bidirectional attention, allowing each token to attend to both preceding and subsequent tokens.

### Classification Head

The contextual token representations are averaged over non-padding positions to obtain a single sentence representation.

This representation is passed through a feedforward classifier:

```text
Encoder output
      |
Masked mean pooling
      |
Linear (64 → 100)
      |
     ReLU
      |
Linear (100 → 3)
      |
Class logits
```

The final layer produces scores for the three speaker classes.

## 6. Part 1 Hyperparameters

| Hyperparameter | Value |
|---|---|
| Vocabulary size | 5,634 |
| Embedding dimension | 64 |
| Attention heads | 2 |
| Transformer blocks | 4 |
| Feedforward hidden dimension | 100 |
| Classifier hidden dimension | 100 |
| Maximum sequence length | 32 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Training epochs | 15 |
| Random seed | 42 |

### Model Parameters

- Encoder: **482,192**
- Encoder + classifier: **488,995**

## 7. Running Part 1

From the repository root:

```bash
python -u PA2_code/main.py --part part1
```

Alternatively, from inside `PA2_code`:

```bash
python -u main.py --part part1
```

The script automatically:

1. Loads the classification training and test datasets.
2. Builds the word-level vocabulary from training text.
3. Initializes the Transformer encoder and classifier.
4. Trains both components jointly for 15 epochs.
5. Evaluates test accuracy after every epoch.
6. Saves training metrics and accuracy figures.
7. Performs attention-normalization sanity checks.
8. Generates attention heatmaps.
9. Saves the final model parameters.

## 8. Part 1 Results

| Epoch | Training Loss | Test Accuracy |
|---|---:|---:|
| 1 | 1.0608 | 46.53% |
| 2 | 0.9989 | 42.80% |
| 3 | 0.9063 | 59.73% |
| 4 | 0.7778 | 64.40% |
| 5 | 0.5989 | 70.13% |
| 6 | 0.4433 | 76.27% |
| 7 | 0.2742 | 79.60% |
| 8 | 0.1817 | 83.47% |
| 9 | 0.1376 | 83.60% |
| 10 | 0.1167 | 85.33% |
| 11 | 0.0650 | 86.13% |
| 12 | 0.0555 | 85.60% |
| 13 | 0.0631 | 84.00% |
| 14 | 0.0391 | 83.73% |
| 15 | 0.0198 | 83.87% |

**Final test accuracy: 83.87%**

**Highest observed test accuracy: 86.13% at epoch 11**

Accuracy generally improves during training, reaching its highest observed value at epoch 11.

Although training loss continues to decrease after epoch 11, test accuracy fluctuates and ends at 83.87%. This suggests some overfitting or generalization instability.

The final result is reported after the full prescribed 15 epochs.

### Attention Sanity Checks

The attention-normalization checks verify that attention probabilities sum to approximately one for every query position, head, and layer.

The maximum observed deviations were:

- Sentence 1: `1.79e-7`
- Sentence 2: `2.38e-7`

Both checks passed.

The encoder attention maps demonstrate bidirectional access to context.

### Generated Part 1 Files

```text
outputs/part1/
├── epoch_metrics.csv
├── accuracy_curve.png
├── sentence_1_attention.png
├── sentence_2_attention.png
├── summary.json
├── vocabulary.json
└── classifier_final.pt
```

---

# Part 2 — Causal Transformer Decoder

## 9. Objective

Part 2 implements a word-level autoregressive language model using a custom Transformer decoder.

The model is trained to predict the next token given the preceding tokens.

Unlike the encoder, the decoder uses causal self-attention, preventing access to future tokens.

## 10. Decoder Architecture

The decoder consists of:

- Learned token embeddings
- Learned absolute position embeddings
- Four causal Transformer blocks
- Two attention heads per block
- Feedforward hidden size of 100
- ReLU activation
- Residual connections and layer normalization
- A final vocabulary projection

The model uses causal masking such that position `i` may attend only to positions `j <= i`.

The output prediction is produced using an untied linear language-modeling head.

The model does not use a pretrained decoder or library Transformer blocks.

### Model Parameters

| Component | Parameters |
|---|---:|
| Decoder backbone | 482,192 |
| Language-model head | 366,210 |
| Total | 848,402 |

## 11. Part 2 Hyperparameters

| Hyperparameter | Value |
|---|---|
| Vocabulary size | 5,634 |
| Embedding dimension | 64 |
| Attention heads | 2 |
| Transformer blocks | 4 |
| Feedforward hidden dimension | 100 |
| Batch size | 16 |
| Sequence length | 32 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Training iterations | 500 |
| Evaluation interval | 100 |
| Evaluation batch limit | 200 |
| Random seed | 42 |

The training process performs 500 optimizer updates over batches of 16 sequences of 32 target positions.

This corresponds to:

**500 × 16 × 32 = 256,000 target-token positions**

Because the dataset uses overlapping sliding windows, these positions are not all unique corpus tokens.

## 12. Running Part 2

From the repository root:

```bash
python -u PA2_code/main.py --part part2
```

Alternatively:

```bash
cd PA2_code
python -u main.py --part part2
```

The script:

1. Loads the training text and three evaluation corpora.
2. Initializes the causal Transformer decoder.
3. Checks future-token prediction invariance.
4. Trains for 500 iterations.
5. Computes perplexity every 100 iterations.
6. Evaluates the three held-out politician datasets.
7. Saves perplexity metrics and curves.
8. Generates causal attention maps.
9. Performs attention-normalization and masking checks.
10. Saves the final decoder checkpoint.

## 13. Evaluation Metric

The evaluation uses perplexity:

```text
Perplexity = exp(mean token cross-entropy)
```

Lower perplexity indicates that the language model assigns higher probability to the observed next tokens.

Perplexity is evaluated on:

- `test_LM_obama.txt`
- `test_LM_wbush.txt`
- `test_LM_hbush.txt`

The implementation evaluates up to the first 200 batches of sliding windows per corpus.

Each full 200-batch evaluation covers 102,400 target positions, including repeated tokens from overlapping windows.

These are fixed-prefix sliding-window perplexity estimates rather than unique-token, full-file perplexities.

## 14. Part 2 Results

| Iteration | Train PPL | Obama PPL | W. Bush PPL | H. W. Bush PPL |
|---|---:|---:|---:|---:|
| 100 | 621.46 | 700.45 | 808.06 | 708.30 |
| 200 | 481.78 | 606.45 | 674.12 | 593.14 |
| 300 | 340.80 | 490.26 | 565.18 | 489.01 |
| 400 | 237.09 | 422.79 | 488.62 | 421.47 |
| 500 | 183.19 | 391.65 | 472.84 | 404.10 |

### Final Perplexity

- Training: **183.19**
- Barack Obama: **391.65**
- George W. Bush: **472.84**
- George H. W. Bush: **404.10**

All evaluated perplexities decrease across training.

George W. Bush's test corpus produces the highest perplexity, indicating that the model finds it more difficult under this evaluation protocol.

Potential contributing factors include lexical distribution, speech style, unknown-token frequency, and similarity to the training data. The aggregate perplexities alone do not establish which factor is responsible.

The remaining train-test gap suggests that the model fits the training distribution better than the evaluated held-out speeches.

### Decoder Sanity Checks

Two checks validate causal behavior.

**Attention normalization:** Every attention row sums to approximately one.

**Causal masking:** Attention entries corresponding to future tokens are zero.

An additional whole-model check changes the last token of an input sequence and verifies that earlier output predictions remain unchanged.

The future-token prediction test passed before and after training.

The attention plots also display the expected lower-triangular causal structure.

### Generated Part 2 Files

```text
outputs/part2/
├── perplexity_metrics.csv
├── perplexity_curve.png
├── sentence_1_attention.png
├── sentence_2_attention.png
├── summary.json
├── vocabulary.json
└── decoder_final.pt
```

---

# Part 3 — Dissecting a Pretrained Language Model

## 15. Objective

Part 3 investigates how the internal representations of a pretrained language model change throughout pretraining.

The experiments use:

**Model:** `EleutherAI/pythia-160m-deduped`

The analysis focuses on:

1. Research findings from the Pythia paper.
2. Predictions recovered from intermediate hidden states using the logit lens.
3. Changes in layer readability across training checkpoints.
4. Compatibility between neural network components trained at different stages.

The implementation uses Hugging Face Transformers to load pretrained checkpoints and custom code for analysis.

## 16. Model Configuration

| Parameter | Value |
|---|---|
| Model | Pythia-160M Deduped |
| Transformer layers | 12 |
| Attention heads | 12 |
| Hidden dimension | 768 |
| Model weights | float32 |
| Evaluation context | 128 tokens |
| Random seed | 42 |

Part 3 uses the pretrained Pythia tokenizer instead of the word-level tokenizer used in Parts 1 and 2.

The saved run tokenizes the Obama evaluation corpus into:

- 6,017 Pythia tokens
- 6,016 next-token transitions

The model evaluates every adjacent token pair once, using context windows of up to 128 prediction positions. Context is reset at window boundaries.

## 17. Running Part 3

The complete experiment is implemented in:

```text
PA2_code/CSE256_PA2_Part3_FINAL.ipynb
```

From the repository root:

```bash
jupyter notebook PA2_code/CSE256_PA2_Part3_FINAL.ipynb
```

Execute all notebook cells sequentially.

The notebook automatically creates an output directory at:

```text
PA2_code/outputs/part3/
```

**Important:** Part 3 downloads multiple Pythia checkpoints, each requiring substantial disk space. Internet connectivity is necessary for the initial downloads.

The saved execution results are embedded in the submitted notebook. Reproduction requires the corresponding dataset, dependencies, and pretrained checkpoints.

## 18. Part 3.1 — Reading Pythia

The notebook discusses the paper:

Biderman et al., *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*, ICML 2023.

The two reading questions address:

- Why releasing multiple checkpoints and preserving training-data order is useful for studying model behavior.
- Findings from comparing deduplicated and non-deduplicated Pythia models.

Written answers and paper-section references are included directly in the notebook.

## 19. Part 3.2 — Logit Lens

The logit lens applies the model's final output readout to intermediate hidden states.

This allows examination of the token predictions recoverable at different depths of the network.

The analysis includes the embedding representation (layer 0) and the outputs of all 12 Transformer layers.

### Final Checkpoint Analysis

The final checkpoint is:

```text
step143000
```

Two sentences are examined:

**Sentence 1:**

"They have not served a Red America or a Blue America"

**Sentence 2:**

"The curious student opened the book and found a surprising answer."

For each sentence, the notebook produces a layer-by-position grid showing:

- The top predicted next token at each layer and position.
- The probability assigned to the actual next token.

### Accuracy Across Layers

The notebook evaluates the logit lens over the entire Obama test corpus.

Selected results at the final checkpoint:

| Layer | Correct Predictions | Top-1 Accuracy |
|---|---:|---:|
| 0 — Embeddings | 1 / 6,016 | 0.016622% |
| 6 — Middle layer | 283 / 6,016 | 4.704122% |
| 12 — Final layer | 2,086 / 6,016 | 34.674202% |

The final layer provides substantially higher accuracy than the embedding or middle-layer readouts.

However, accuracy does not improve monotonically at every intermediate layer.

### Across Pretraining Checkpoints

The experiment compares four checkpoints:

```text
step512
step1000
step8000
step143000
```

Accuracy-by-layer curves are plotted together.

Final-layer accuracy increases as training progresses.

However, the final checkpoint's intermediate layers are not always the most readable using the untuned logit lens.

For example, layer 6 achieves:

| Checkpoint | Layer 6 Accuracy |
|---|---:|
| step512 | 11.4860% |
| step1000 | 15.0432% |
| step8000 | 14.9435% |
| step143000 | 4.7041% |

This suggests that intermediate representations may become less aligned with the final output readout during training.

A lower intermediate logit-lens accuracy does not necessarily imply that the intermediate layer has lost predictive information.

The detailed discussion is included in the executed notebook.

## 20. Part 3.3 — Layer Transplant Experiments

The transplantation experiment measures the compatibility of Transformer components across pretraining checkpoints.

The recipient is always the intact final model:

```text
step143000
```

Donor checkpoints:

```text
step0
step1000
step8000
step32000
```

The experiment replaces one component at a time:

- Token embedding layer
- Each of the 12 Transformer blocks
- Output unembedding layer

After each replacement, the model's test cross-entropy is evaluated on the Obama test corpus.

The original weights are restored before the next transplant.

### Baseline

The intact final model has a baseline test loss of:

**3.4617656152 nats/token**

For each transplant, the change in loss is computed as:

```text
Delta loss =
    Transplanted model loss - Intact model loss
```

A positive change indicates worse performance than the intact final model.

### Main Observations

The experiments show that:

- The output unembedding is particularly sensitive to replacement.
- The first and final Transformer blocks are also relatively sensitive.
- Several middle blocks tolerate earlier checkpoint weights with smaller losses.
- A later donor checkpoint does not always result in a smaller performance degradation.

For example, replacing block 5 with weights from step1000 increases loss by approximately 0.526555 nats/token, while replacement from step8000 increases loss by approximately 3.257457 nats/token.

The heatmap displays all 56 component-donor combinations with exact numerical annotations and logarithmically scaled colors.

The notebook includes a discussion of these effects.

## 21. Generated Part 3 Files

```text
outputs/part3/
├── run_config.json
├── baseline.json
├── final_accuracy.csv
├── final_accuracy.png
├── checkpoint_accuracy.csv
├── checkpoint_accuracy.png
├── grid_obama.png
├── grid_obama.npz
├── grid_original.png
├── grid_original.npz
├── transplant_deltas.csv
├── transplant_losses.csv
└── transplant_heatmap.png
```

The CSV and NumPy files contain the computed numerical results used to produce the figures.

The executed notebook remains the primary submission artifact for Part 3.

---

## 22. Reproducibility

The implementation uses fixed random seeds where applicable.

Parts 1 and 2 use:

```python
SEED = 42
```

The tokenizer vocabulary is constructed from training files only.

The models use deterministic seed initialization, and the scripts save their numerical outputs for inspection.

Some differences may occur across hardware, library versions, and execution environments.

For Part 3:

- All checkpoints use the same evaluation corpus.
- The same tokenization and context-window protocol is used.
- Model weights are loaded in float32.
- Transplant experiments restore the intact recipient weights after each test.
- The original baseline is checked again after each donor checkpoint.

The Part 3 Pythia-token metrics should not be compared directly with the Part 2 word-level perplexity results because the tokenization and evaluation protocols differ.

---

## 23. Assignment Report

The accompanying report contains two parts.

### Part A — Encoder and Decoder

Approximately five pages covering:

- Architecture and implementation
- Hyperparameters
- Parameter counts
- Classification accuracy over 15 epochs
- Language-model perplexities over 500 iterations
- Attention-map visualizations
- Sanity checks and interpretation

### Part B — Pretrained Pythia Analysis

A PDF export of the executed Part 3 notebook containing:

- Pythia paper answers
- Logit-lens prediction grids
- Accuracy-by-layer measurements
- Across-checkpoint comparisons
- Transplant heatmap
- Numerical outputs
- Markdown explanations and conclusions

Part B is appended after Part A in the combined report PDF.

---

## 24. Gradescope Submission

The assignment requires the following deliverables.

### Code Submission

Include:

- `main.py`
- `transformer.py`
- `dataset.py`
- `tokenizer.py`
- `utilities.py`
- The fully executed Part 3 `.ipynb`
- `README.md`
- Required supporting files and dependency information

### Report Submission

Submit a single combined PDF containing:

1. Part A — Parts 1 and 2, approximately five pages.
2. Part B — PDF export of the executed Part 3 notebook.

The Part 3 notebook and its PDF export must correspond to the same saved execution, including all numerical results, figures, and explanations.

All requested Part 3 computations must remain visible in the executed notebook.

---

## 25. References

1. Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.
2. Biderman et al. (2023). *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*. ICML.
3. nostalgebraist (2020). *Interpreting GPT: The Logit Lens*.
4. Belrose et al. (2023). *Eliciting Latent Predictions from Transformers with the Tuned Lens*.

---

**Author:** Jai Dayanand  
**Course:** CSE 256 — Natural Language Processing  
**Institution:** University of California, San Diego  
**Quarter:** Fall 2026
