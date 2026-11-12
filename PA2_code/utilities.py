from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch


class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(
        self,
        sentence,
        block_size,
        output_dir,
        prefix="sentence",
        layer=0,
        head=0,
        causal=False,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        word_ids = self.tokenizer.encode(sentence)[:block_size]

        if not word_ids:
            raise ValueError("The sentence must contain tokens.")

        device = next(self.model.parameters()).device

        input_tensor = torch.tensor(
            [word_ids],
            dtype=torch.long,
            device=device,
        )

        was_training = self.model.training
        self.model.eval()

        try:
            with torch.no_grad():
                _, attention_maps = self.model(input_tensor)

            max_row_error = 0.0

            for layer_index, attention in enumerate(attention_maps):
                if not torch.isfinite(attention).all():
                    raise AssertionError(
                        f"Non-finite attention in layer {layer_index + 1}."
                    )

                row_sums = attention.sum(dim=-1)

                error = (row_sums - 1.0).abs().max().item()
                max_row_error = max(max_row_error, error)

                if not torch.allclose(
                    row_sums,
                    torch.ones_like(row_sums),
                    atol=1e-5,
                    rtol=1e-5,
                ):
                    raise AssertionError(
                        f"Attention normalization failed in "
                        f"layer {layer_index + 1}."
                    )

                if causal:
                    future_attention = torch.triu(
                        attention,
                        diagonal=1,
                    )

                    if future_attention.abs().max().item() > 1e-7:
                        raise AssertionError(
                            "Causal attention includes future tokens."
                        )

            print(
                f"{prefix}: attention normalization PASSED | "
                f"maximum row-sum error = {max_row_error:.2e}"
            )

            matrix = (
                attention_maps[layer][0, head]
                .detach()
                .cpu()
                .numpy()
            )

            labels = [
                self.tokenizer.itos[index]
                for index in word_ids
            ]

            fig, ax = plt.subplots(figsize=(9, 7))

            image = ax.imshow(
                matrix,
                cmap="viridis",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )

            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))

            ax.set_xticklabels(
                labels,
                rotation=60,
                ha="right",
            )
            ax.set_yticklabels(labels)

            ax.set_xlabel("Key token: attended to")
            ax.set_ylabel("Query token: attending")

            ax.set_title(
                f"Layer {layer + 1}, head {head + 1}\n{sentence}"
            )

            fig.colorbar(
                image,
                ax=ax,
                label="Attention probability",
            )

            fig.tight_layout()

            figure_path = output_dir / f"{prefix}_attention.png"

            fig.savefig(
                figure_path,
                dpi=200,
                bbox_inches="tight",
            )

            plt.close(fig)

            print(f"Saved: {figure_path}")

        finally:
            self.model.train(was_training)