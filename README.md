# LLM-Capacity

Implementation is referenced from [nano-gpt](https://github.com/karpathy/nanoGPT/tree/master) by Karpathy
- Note : Use [uv](https://docs.astral.sh/uv/getting-started/installation/) for managing dependencies and environments

## Prepare data

- Synthetic Data

```bash
uv run src/synthetic/prepare.py
```

- OpenWebText dataset

```bash
uv run src/openwebtext/prepare.py
```

## Train with synthetic data

- Note: Use 4 nodes for parallel training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 src/train_synthetic.py
```

# Evaluate Capacity
- Change `ckpt_path`to the model you want to evaluate 

```bash
uv run src/evaluate.py
```