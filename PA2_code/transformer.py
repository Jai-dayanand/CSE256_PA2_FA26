import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, causal=False):
        super().__init__()

        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.causal = causal

        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.output_projection = nn.Linear(n_embd, n_embd)

    def forward(self, x, valid_mask=None):
        batch_size, seq_len, n_embd = x.shape

        def split_heads(tensor):
            tensor = tensor.reshape(
                batch_size,
                seq_len,
                self.n_head,
                self.head_dim,
            )
            return tensor.transpose(1, 2)

        q = split_heads(self.query(x))
        k = split_heads(self.key(x))
        v = split_heads(self.value(x))

        # Shape: (batch, heads, query positions, key positions).
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        if valid_mask is not None:
            # Exclude padding tokens as attention keys.
            scores = scores.masked_fill(
                ~valid_mask[:, None, None, :],
                float("-inf"),
            )

        if self.causal:
            future_mask = torch.triu(
                torch.ones(
                    seq_len,
                    seq_len,
                    dtype=torch.bool,
                    device=x.device,
                ),
                diagonal=1,
            )

            scores = scores.masked_fill(
                future_mask,
                float("-inf"),
            )

        attention = F.softmax(scores, dim=-1)

        output = attention @ v

        # Combine heads back into one embedding dimension.
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, n_embd)

        output = self.output_projection(output)

        return output, attention


class FeedForward(nn.Module):
    def __init__(self, n_embd, hidden_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_embd, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_embd),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        hidden_size,
        causal=False,
    ):
        super().__init__()

        self.ln_attention = nn.LayerNorm(n_embd)

        self.attention = MultiHeadSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            causal=causal,
        )

        self.ln_feedforward = nn.LayerNorm(n_embd)

        self.feedforward = FeedForward(
            n_embd=n_embd,
            hidden_size=hidden_size,
        )

    def forward(self, x, valid_mask=None):
        attention_output, attention_map = self.attention(
            self.ln_attention(x),
            valid_mask=valid_mask,
        )

        x = x + attention_output
        x = x + self.feedforward(self.ln_feedforward(x))

        return x, attention_map


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=32,
        n_embd=64,
        n_head=2,
        n_layer=4,
        ff_hidden=100,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd,
            padding_idx=0,
        )

        self.position_embedding = nn.Embedding(
            block_size,
            n_embd,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    hidden_size=ff_hidden,
                    causal=False,
                )
                for _ in range(n_layer)
            ]
        )

        self.final_norm = nn.LayerNorm(n_embd)

    def forward(self, input_ids):
        _, seq_len = input_ids.shape

        if seq_len > self.block_size:
            raise ValueError("Input sequence exceeds block_size.")

        valid_mask = input_ids.ne(0)

        if not valid_mask.any(dim=1).all():
            raise ValueError(
                "Each input must contain a non-padding token."
            )

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        )

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)[None, :, :]
        )

        attention_maps = []

        for block in self.blocks:
            x, attention = block(
                x,
                valid_mask=valid_mask,
            )
            attention_maps.append(attention)

        return self.final_norm(x), attention_maps


class SpeakerClassifier(nn.Module):
    def __init__(
        self,
        encoder,
        n_embd=64,
        n_hidden=100,
        n_output=3,
    ):
        super().__init__()

        self.encoder = encoder

        self.classifier = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, input_ids):
        token_vectors, _ = self.encoder(input_ids)

        # Average real word vectors, excluding padding.
        valid_mask = input_ids.ne(0).unsqueeze(-1)

        summed_vectors = (
            token_vectors * valid_mask
        ).sum(dim=1)

        token_counts = valid_mask.sum(dim=1).clamp_min(1)

        mean_vector = summed_vectors / token_counts

        # Cross-entropy expects raw class scores.
        return self.classifier(mean_vector)
    
class TransformerDecoder(nn.Module):
    """
    Word-level causal language model.

    Returns:
        logits: (batch, sequence_length, vocabulary_size)
        attention_maps: one tensor per Transformer layer
    """

    def __init__(
        self,
        vocab_size,
        block_size=32,
        n_embd=64,
        n_head=2,
        n_layer=4,
        ff_hidden=100,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd,
        )

        self.position_embedding = nn.Embedding(
            block_size,
            n_embd,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    hidden_size=ff_hidden,
                    causal=True,
                )
                for _ in range(n_layer)
            ]
        )

        self.final_norm = nn.LayerNorm(n_embd)

        # Separate, untied output projection.
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, input_ids):
        _, seq_len = input_ids.shape

        if seq_len > self.block_size:
            raise ValueError("Input sequence exceeds block_size.")

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        )

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)[None, :, :]
        )

        attention_maps = []

        for block in self.blocks:
            x, attention = block(x)
            attention_maps.append(attention)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, attention_maps