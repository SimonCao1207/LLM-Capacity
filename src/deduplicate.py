# https://github.com/huggingface/datatrove/blob/main/examples/fineweb.py

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -- Hashing configuration
@dataclass(frozen=True)
class HashConfig:
    precision: Literal[32, 64] = 64
    hash_fc: Literal["sha1", "xxhash"] = "sha1"

    @property
    def np_dtype(self):
        return np.uint32 if self.precision == 32 else np.uint64

    @property
    def max(self):
        return np.iinfo(self.np_dtype).max


@dataclass
class MinhashConfig:
    n_grams: int = 5
    num_buckets: int = 14
    hashes_per_bucket: int = 8
    seed: int = 1
    hash_config: HashConfig = field(default_factory=HashConfig)


@dataclass(order=True)
class HashSig:
    sig: tuple[int, ...]
    doc_id: int


class MinhashDedupSignature:
    """Minhash Deduplication: First Pipeline Step

        Compute the minhash signature for each document and write it to disk.

    Args:
        output_folder: output folder
        config: minhash configuration (a MinhashConfig object)
    """

    pass


class InMemorySignatureReader:
    def __init__(self, docs: list[list[int]], config: MinhashConfig):
        self.docs = docs
        self.config = config
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= len(self.docs):
            raise StopIteration
        tokens = self.docs[self.current]
        sig = compute_minhash_signature(tokens, self.config)
        item = HashSig(
            sig=tuple(sig),
            doc_id=self.current,
        )
        self.current += 1
        return item


def compute_ngrams(tokens: list[int], n: int) -> list[str]:
    return [" ".join(map(str, tokens[i : i + n])) for i in range(len(tokens) - n + 1)]


def hash_ngram(ngram: str, config: HashConfig) -> int:
    h = hashlib.sha1(ngram.encode("utf-8")).digest()
    dtype = config.np_dtype
    return int(np.frombuffer(h[: dtype().itemsize], dtype=dtype)[0])


def compute_minhash_signature(tokens: list[int], config: MinhashConfig) -> list[int]:
    ngrams = compute_ngrams(tokens, config.n_grams)
    signature = []

    for bucket_id in range(config.num_buckets):
        bucket_hashes = []
        for i in range(config.hashes_per_bucket):
            salted = [f"{bucket_id}_{i}_{ng}" for ng in ngrams]
            hashes = [hash_ngram(ng, config.hash_config) for ng in salted]
            bucket_hashes.append(min(hashes) if hashes else config.hash_config.max)
        signature.extend(bucket_hashes)

    return signature


def deduplicate_sequences(sequences: list[list[int]], config: MinhashConfig) -> list[list[int]]:
    sig_reader = InMemorySignatureReader(sequences, config)
    seen_sigs = set()
    unique_sequences = []

    for sig in sig_reader:
        if sig.sig not in seen_sigs:
            seen_sigs.add(sig.sig)
            unique_sequences.append(sequences[sig.doc_id])

    return unique_sequences


if __name__ == "__main__":
    # Demo
    minhash_config = MinhashConfig(hash_config=HashConfig(hash_fc="sha1", precision=64))
    dummy_data = [list(range(i, i + 64)) for i in range(10)] + [list(range(0, 64))] + [list(range(3, 67))]
    deduped = deduplicate_sequences(dummy_data, minhash_config)
    assert len(deduped) == 10, "Deduplication failed"
