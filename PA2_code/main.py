import argparse
import csv
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from dataset import (
    SpeechesClassificationDataset,
    LanguageModelingDataset,
)
from tokenizer import SimpleTokenizer
from transformer import (
    TransformerEncoder,
    SpeakerClassifier,
    TransformerDecoder,
)
from utilities import Utilities


# ============================================================
# Assignment hyperparameters
# ============================================================

SEED = 42

BATCH_SIZE = 16
BLOCK_SIZE = 32
LEARNING_RATE = 1e-3

N_EMBD = 64
N_HEAD = 2
N_LAYER = 4

FF_HIDDEN = 100
CLASSIFIER_HIDDEN = 100
N_OUTPUT = 3

EPOCHS_CLS = 15

MAX_ITERS = 500
EVAL_INTERVAL = 100
EVAL_ITERS = 200


# Paths are based on main.py, regardless of terminal location.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "speechesdataset"


# ============================================================
# Shared helpers
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_tokenizer():
    """
    Match the starter's vocabulary-building approach:
    use both training files and exclude all test files.
    """
    training_files = [
        DATA_DIR / "train_CLS.tsv",
        DATA_DIR / "train_LM.txt",
    ]

    texts = [
        path.read_text(encoding="utf-8")
        for path in training_files
    ]

    return SimpleTokenizer("\n".join(texts))


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def save_json(path, data):
    Path(path).write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# Part 1: Encoder and speaker classifier
# ============================================================

def collate_batch(batch):
    sequences, labels = zip(*batch)

    sequences = [
        sequence[:BLOCK_SIZE]
        for sequence in sequences
    ]

    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0,
    )

    if padded.shape[1] < BLOCK_SIZE:
        padded = F.pad(
            padded,
            (0, BLOCK_SIZE - padded.shape[1]),
            value=0,
        )

    return padded, torch.stack(labels)


@torch.no_grad()
def compute_classifier_accuracy(model, data_loader, device):
    was_training = model.training
    model.eval()

    correct = 0
    total = 0

    try:
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            predictions = model(inputs).argmax(dim=-1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.numel()

    finally:
        model.train(was_training)

    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    return 100.0 * correct / total


def save_accuracy_plot(history, output_dir):
    epochs = [
        row["epoch"]
        for row in history
    ]

    accuracies = [
        row["test_accuracy_percent"]
        for row in history
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epochs,
        accuracies,
        marker="o",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Part 1: speaker classification")
    ax.set_xticks(epochs)
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "accuracy_curve.png",
        dpi=200,
    )

    plt.close(fig)


def run_part1(device):
    set_seed(SEED)

    output_dir = BASE_DIR / "outputs" / "part1"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Loading data and building tokenizer...",
        flush=True,
    )

    tokenizer = build_tokenizer()

    train_dataset = SpeechesClassificationDataset(
        tokenizer,
        DATA_DIR / "train_CLS.tsv",
    )

    test_dataset = SpeechesClassificationDataset(
        tokenizer,
        DATA_DIR / "test_CLS.tsv",
    )

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        generator=generator,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )

    encoder = TransformerEncoder(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        ff_hidden=FF_HIDDEN,
    )

    model = SpeakerClassifier(
        encoder=encoder,
        n_embd=N_EMBD,
        n_hidden=CLASSIFIER_HIDDEN,
        n_output=N_OUTPUT,
    ).to(device)

    encoder_parameters = count_parameters(model.encoder)
    total_parameters = count_parameters(model)

    print(f"Device: {device}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    print(f"Encoder parameters: {encoder_parameters:,}")
    print(
        f"Encoder + classifier parameters: "
        f"{total_parameters:,}"
    )
    print(flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []
    csv_path = output_dir / "epoch_metrics.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "test_accuracy_percent",
            ],
        )

        writer.writeheader()

        for epoch in range(1, EPOCHS_CLS + 1):
            model.train()

            loss_sum = 0.0
            example_count = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)

                logits = model(inputs)

                loss = F.cross_entropy(
                    logits,
                    labels,
                )

                loss.backward()
                optimizer.step()

                batch_count = labels.size(0)

                loss_sum += loss.item() * batch_count
                example_count += batch_count

            train_loss = loss_sum / example_count

            test_accuracy = compute_classifier_accuracy(
                model,
                test_loader,
                device,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_accuracy_percent": test_accuracy,
            }

            history.append(row)
            writer.writerow(row)
            file.flush()

            print(
                f"Epoch {epoch:02d}/{EPOCHS_CLS} | "
                f"train loss {train_loss:.4f} | "
                f"test accuracy {test_accuracy:.2f}%",
                flush=True,
            )

    save_accuracy_plot(history, output_dir)

    config = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "block_size": BLOCK_SIZE,
        "learning_rate": LEARNING_RATE,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "encoder_ff_hidden": FF_HIDDEN,
        "classifier_hidden": CLASSIFIER_HIDDEN,
        "epochs": EPOCHS_CLS,
    }

    summary = {
        "config": config,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "vocab_size": tokenizer.vocab_size,
        "encoder_parameters": encoder_parameters,
        "total_parameters": total_parameters,
        "final_test_accuracy_percent": history[-1][
            "test_accuracy_percent"
        ],
    }

    save_json(
        output_dir / "summary.json",
        summary,
    )

    save_json(
        output_dir / "vocabulary.json",
        tokenizer.stoi,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "vocab_size": tokenizer.vocab_size,
            "stoi": tokenizer.stoi,
        },
        output_dir / "classifier_final.pt",
    )

    utility = Utilities(
        tokenizer,
        model.encoder,
    )

    sentences = [
        "They have not served a Red America or a Blue America",
        "We must work together to build a better future.",
    ]

    for index, sentence in enumerate(sentences, start=1):
        utility.sanity_check(
            sentence=sentence,
            block_size=BLOCK_SIZE,
            output_dir=output_dir,
            prefix=f"sentence_{index}",
            layer=N_LAYER - 1,
            head=0,
        )

    print()
    print(
        f"Final test accuracy: "
        f"{summary['final_test_accuracy_percent']:.2f}%"
    )
    print(f"Outputs saved in: {output_dir}")


# ============================================================
# Part 2: Causal decoder language model
# ============================================================

@torch.no_grad()
def compute_lm_perplexity(
    model,
    data_loader,
    device,
    max_batches=EVAL_ITERS,
):
    """
    Evaluate token-weighted cross-entropy, then exponentiate.

    Uses up to max_batches batches of the supplied loader.
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    try:
        for batch_index, (inputs, targets) in enumerate(data_loader):
            if batch_index >= max_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model(inputs)

            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )

            total_loss += loss_sum.item()
            total_tokens += targets.numel()

    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise ValueError("No tokens were evaluated.")

    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss)

    return perplexity, mean_loss, total_tokens


def save_perplexity_plot(history, output_dir):
    fig, ax = plt.subplots(figsize=(9, 6))

    iterations = [
        row["iteration"]
        for row in history
    ]

    for name in ["train", "obama", "wbush", "hbush"]:
        perplexities = [
            row[f"{name}_perplexity"]
            for row in history
        ]

        ax.plot(
            iterations,
            perplexities,
            marker="o",
            label=name,
        )

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Perplexity")
    ax.set_title("Part 2: language modeling perplexity")
    ax.set_xticks(iterations)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "perplexity_curve.png",
        dpi=200,
    )

    plt.close(fig)


@torch.no_grad()
def check_causal_predictions(model, tokenizer, device):
    """
    Changing the final input token must not change predictions
    at earlier positions.
    """
    was_training = model.training
    model.eval()

    try:
        ids = tokenizer.encode(
            "We must work together to build a better future."
        )[:BLOCK_SIZE]

        if len(ids) < 2:
            raise ValueError(
                "Causality check requires at least two tokens."
            )

        original = torch.tensor(
            [ids],
            dtype=torch.long,
            device=device,
        )

        modified = original.clone()

        modified[0, -1] = (
            modified[0, -1] + 1
        ) % tokenizer.vocab_size

        original_logits, _ = model(original)
        modified_logits, _ = model(modified)

        if not torch.allclose(
            original_logits[:, :-1],
            modified_logits[:, :-1],
            atol=1e-5,
            rtol=1e-5,
        ):
            raise AssertionError(
                "Causality failed: a future token changed "
                "earlier predictions."
            )

        print(
            "Future-token prediction check: PASSED",
            flush=True,
        )

    finally:
        model.train(was_training)


def run_part2(device):
    set_seed(SEED)

    output_dir = BASE_DIR / "outputs" / "part2"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Loading language modeling data...",
        flush=True,
    )

    tokenizer = build_tokenizer()

    corpus_paths = {
        "train": DATA_DIR / "train_LM.txt",
        "obama": DATA_DIR / "test_LM_obama.txt",
        "wbush": DATA_DIR / "test_LM_wbush.txt",
        "hbush": DATA_DIR / "test_LM_hbush.txt",
    }

    datasets = {}

    for name, path in corpus_paths.items():
        text = path.read_text(encoding="utf-8")

        datasets[name] = LanguageModelingDataset(
            tokenizer,
            text,
            BLOCK_SIZE,
        )

    # Separate generators keep evaluation from changing
    # the training batch order.
    train_generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=train_generator,
        num_workers=0,
    )

    if len(train_loader) == 0:
        raise ValueError(
            "Not enough training windows for one full batch."
        )

    eval_loaders = {}

    for index, (name, dataset) in enumerate(datasets.items()):
        eval_generator = torch.Generator().manual_seed(
            SEED + 100 + index
        )

        eval_loaders[name] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            generator=eval_generator,
            num_workers=0,
        )

    model = TransformerDecoder(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        ff_hidden=FF_HIDDEN,
    ).to(device)

    total_parameters = count_parameters(model)
    head_parameters = count_parameters(model.lm_head)
    backbone_parameters = (
        total_parameters - head_parameters
    )

    print(f"Device: {device}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(
        f"Decoder backbone parameters: "
        f"{backbone_parameters:,}"
    )
    print(f"LM head parameters: {head_parameters:,}")
    print(
        f"Total decoder LM parameters: "
        f"{total_parameters:,}"
    )

    for name, dataset in datasets.items():
        evaluation_batches = min(
            EVAL_ITERS,
            len(eval_loaders[name]),
        )

        print(
            f"{name}: {len(dataset):,} sliding windows; "
            f"evaluating up to {evaluation_batches} batches"
        )

    print(flush=True)

    check_causal_predictions(
        model,
        tokenizer,
        device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []

    fieldnames = [
        "iteration",
        "train_perplexity",
        "obama_perplexity",
        "wbush_perplexity",
        "hbush_perplexity",
        "train_mean_loss",
        "obama_mean_loss",
        "wbush_mean_loss",
        "hbush_mean_loss",
        "train_eval_tokens",
        "obama_eval_tokens",
        "wbush_eval_tokens",
        "hbush_eval_tokens",
    ]

    train_iterator = iter(train_loader)

    interval_loss_sum = 0.0
    interval_tokens = 0
    training_target_tokens = 0

    csv_path = output_dir / "perplexity_metrics.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        model.train()

        for iteration in range(1, MAX_ITERS + 1):
            # Restart the loader if needed, ensuring exactly
            # MAX_ITERS optimizer updates.
            try:
                inputs, targets = next(train_iterator)

            except StopIteration:
                train_iterator = iter(train_loader)
                inputs, targets = next(train_iterator)

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(inputs)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at iteration {iteration}."
                )

            loss.backward()
            optimizer.step()

            token_count = targets.numel()

            interval_loss_sum += loss.item() * token_count
            interval_tokens += token_count
            training_target_tokens += token_count

            if iteration % 25 == 0:
                recent_loss = (
                    interval_loss_sum / interval_tokens
                )

                print(
                    f"Iteration {iteration:03d}/{MAX_ITERS} | "
                    f"recent training loss {recent_loss:.4f}",
                    flush=True,
                )

                interval_loss_sum = 0.0
                interval_tokens = 0

            if (
                iteration % EVAL_INTERVAL == 0
                or iteration == MAX_ITERS
            ):
                print(
                    "Evaluating perplexities...",
                    flush=True,
                )

                row = {
                    "iteration": iteration,
                }

                for name, loader in eval_loaders.items():
                    perplexity, mean_loss, evaluated_tokens = (
                        compute_lm_perplexity(
                            model,
                            loader,
                            device,
                            max_batches=EVAL_ITERS,
                        )
                    )

                    row[f"{name}_perplexity"] = perplexity
                    row[f"{name}_mean_loss"] = mean_loss
                    row[f"{name}_eval_tokens"] = evaluated_tokens

                history.append(row)
                writer.writerow(row)
                file.flush()

                print(
                    f"Iteration {iteration:03d} | "
                    f"train PPL "
                    f"{row['train_perplexity']:.2f} | "
                    f"Obama "
                    f"{row['obama_perplexity']:.2f} | "
                    f"W. Bush "
                    f"{row['wbush_perplexity']:.2f} | "
                    f"H. W. Bush "
                    f"{row['hbush_perplexity']:.2f}",
                    flush=True,
                )

    save_perplexity_plot(
        history,
        output_dir,
    )

    config = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "block_size": BLOCK_SIZE,
        "learning_rate": LEARNING_RATE,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "ff_hidden": FF_HIDDEN,
        "max_iters": MAX_ITERS,
        "eval_interval": EVAL_INTERVAL,
        "eval_iters": EVAL_ITERS,
        "evaluation_protocol": (
            "First up to 200 batches of sliding windows "
            "per corpus; token-weighted cross-entropy, "
            "then exponentiation."
        ),
    }

    summary = {
        "config": config,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "vocab_size": tokenizer.vocab_size,
        "decoder_backbone_parameters": backbone_parameters,
        "lm_head_parameters": head_parameters,
        "total_decoder_lm_parameters": total_parameters,
        "training_target_tokens": training_target_tokens,
        "final_metrics": history[-1],
    }

    save_json(
        output_dir / "summary.json",
        summary,
    )

    save_json(
        output_dir / "vocabulary.json",
        tokenizer.stoi,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "vocab_size": tokenizer.vocab_size,
            "stoi": tokenizer.stoi,
        },
        output_dir / "decoder_final.pt",
    )

    utility = Utilities(
        tokenizer,
        model,
    )

    sentences = [
        "They have not served a Red America or a Blue America",
        "We must work together to build a better future.",
    ]

    for index, sentence in enumerate(sentences, start=1):
        utility.sanity_check(
            sentence=sentence,
            block_size=BLOCK_SIZE,
            output_dir=output_dir,
            prefix=f"sentence_{index}",
            layer=N_LAYER - 1,
            head=0,
            causal=True,
        )

    check_causal_predictions(
        model,
        tokenizer,
        device,
    )

    print()
    print("Part 2 complete.")
    print(
        f"Training target tokens: "
        f"{training_target_tokens:,}"
    )
    print(f"Outputs saved in: {output_dir}")


# ============================================================
# Command-line entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CSE 256 PA2: Transformer encoder and decoder."
    )

    parser.add_argument(
        "--part",
        choices=["part1", "part2"],
        default="part1",
        help="Choose the assignment part to run.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Use CUDA when available, or select a device.",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    else:
        if (
            args.device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA is unavailable. Use --device cpu."
            )

        device = torch.device(args.device)

    if args.part == "part1":
        run_part1(device)

    else:
        run_part2(device)


if __name__ == "__main__":
    main()