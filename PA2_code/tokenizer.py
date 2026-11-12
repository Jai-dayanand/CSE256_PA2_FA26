from nltk.tokenize import word_tokenize


class SimpleTokenizer:
    """Word-level tokenizer with padding and unknown-token support."""

    def __init__(self, text):
        self.build_vocab(text)

    def build_vocab(self, text):
        tokens = word_tokenize(text)

        self.vocab = set(tokens)

        # Reserve 0 for padding and 1 for unknown words.
        # Sorting keeps token IDs consistent across runs.
        self.stoi = {
            word: index
            for index, word in enumerate(
                sorted(self.vocab - {"<pad>", "<unk>"}),
                start=2,
            )
        }

        self.stoi["<pad>"] = 0
        self.stoi["<unk>"] = 1

        self.itos = {
            index: word
            for word, index in self.stoi.items()
        }

        self.vocab_size = len(self.stoi)

    def encode(self, text):
        """Convert text into a list of integer token IDs."""
        tokens = word_tokenize(text)

        return [
            self.stoi.get(token, self.stoi["<unk>"])
            for token in tokens
        ]

    def decode(self, indices):
        """Convert token IDs back into space-separated text."""
        return " ".join(
            self.itos.get(int(index), "<unk>")
            for index in indices
        )