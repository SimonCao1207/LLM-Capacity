# LLM-Capacity
Replication of paper ["How much do language models memorize?"](https://arxiv.org/abs/2505.24832)
- Note : Use [uv](https://docs.astral.sh/uv/getting-started/installation/) for managing dependencies and environments

## Prepare data

- Synthetic Data (~1M sequences of length 64 tokens, uniformedly sampled) 

```bash
uv run data/synthetic/prepare.py
```

- OpenWebText dataset (optional, for real text experiments)

```bash
uv run data/openwebtext/prepare.py
```

## Train with synthetic data

- Note: Use 4 nodes for parallel training
- Increase `max_iters` in `train_synthetic.py` to `1e6` to match original paper's setting

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 src/train_synthetic.py
```

# Evaluate Capacity
- Change `ckpt_path`to the model you want to evaluate (saved in `/out`)

```bash
uv run src/evaluate.py
```

## Reference
- [nano-gpt](https://github.com/karpathy/nanoGPT/tree/master) by Karpathy
