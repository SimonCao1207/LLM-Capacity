import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from deduplicate import HashConfig, MinhashConfig, deduplicate_sequences


class DeduplicatedTextDataset(Dataset):
    def __init__(
        self, hf_dataset_name="allenai/c4", tokenizer_name="gpt2", split="train", max_sequences=1000000, seq_len=64
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.seq_len = seq_len
        self.max_sequences = max_sequences
        self.minhash_config = MinhashConfig(hash_config=HashConfig(hash_fc="sha1", precision=64))

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset = load_dataset(hf_dataset_name, "en", split=split, streaming=True, trust_remote_code=True)
        self.deduped_chunks = self._deduplicate(dataset)

    def _deduplicate(self, stream):
        all_chunks = []
        processed_sequences = 0

        for example in stream:
            if processed_sequences >= self.max_sequences:
                break

            tokens = self.tokenizer(example["text"], truncation=False, return_attention_mask=False)["input_ids"]

            # Chop into non-overlapping 64-token sequences
            for i in range(0, len(tokens) - self.seq_len + 1, self.seq_len):
                chunk = tokens[i : i + self.seq_len]
                if len(chunk) == self.seq_len:
                    all_chunks.append(chunk)
                    processed_sequences += 1

                    if processed_sequences >= self.max_sequences:
                        break

            if processed_sequences % 1000 == 0:
                print(f"Processed {processed_sequences} sequences...")

        print(f"Total chunks collected: {len(all_chunks)}")
        print("Starting deduplication...")

        # Deduplicate using minhash
        unique_chunks = deduplicate_sequences(all_chunks, self.minhash_config)

        print(
            f"Deduplication complete. Unique chunks: {len(unique_chunks)} "
            f"(removed {len(all_chunks) - len(unique_chunks)} duplicates)"
        )

        return unique_chunks

    def __len__(self):
        return len(self.deduped_chunks)

    def __getitem__(self, idx):
        return self.deduped_chunks[idx]


class RandomBitstringDataset(Dataset):
    def __init__(self, dataset_size: int, seq_len: int = 64, vocab_size: int = 2048):
        self.dataset_size = dataset_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        return x[:-1], x[1:]  # input and target


# Example usage
if __name__ == "__main__":
    dataset = DeduplicatedTextDataset(max_sequences=10000)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    print(f"Dataset size after deduplication: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print(f"Sample sequence shape: {len(sample)}")
    print(f"First few tokens: {sample[:10]}")
