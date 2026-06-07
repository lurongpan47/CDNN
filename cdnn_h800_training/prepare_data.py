#!/usr/bin/env python3
"""
=============================================================================
prepare_data.py — Data preparation for CD-Transformer training
=============================================================================

Tokenizes text corpora into binary format for efficient training.
Supports HuggingFace datasets and local text files.

Usage:
    # From HuggingFace dataset
    python prepare_data.py --source hf --dataset openwebtext \
        --tokenizer meta-llama/Llama-2-7b-hf --output ./data

    # From local text files
    python prepare_data.py --source local --input_dir ./corpus \
        --tokenizer gpt2 --output ./data

    # Quick test with small synthetic dataset
    python prepare_data.py --source synthetic --output ./data

Authors: L. Pan (Ainnocence Inc.)
License: MIT
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Optional


def _check_uint16(tokens: np.ndarray, vocab_size: int):
    """Guard against silent truncation: token ids are stored as uint16, so any
    id > 65535 would wrap around and corrupt the data. DeepSeek/Llama vocabs
    (~100k+) exceed this, so fail loudly instead of writing garbage."""
    if vocab_size > 65535:
        raise ValueError(
            f"vocab_size={vocab_size} exceeds uint16 max (65535). "
            f"Token ids would silently wrap. Use uint32 storage (and update "
            f"TokenDataset dtype in train_distributed.py to match).")
    hi = int(tokens.max()) if tokens.size else 0
    if hi > 65535:
        raise ValueError(
            f"Max token id {hi} exceeds uint16 range; refusing to write "
            f"corrupted .bin. Check tokenizer vocab vs uint16 storage.")


def prepare_synthetic(output_dir: str, vocab_size: int = 32000, val_frac: float = 0.05,
                      n_tokens: int = 10_000_000):
    """Generate synthetic token data for testing."""
    print(f"Generating {n_tokens:,} synthetic tokens (vocab={vocab_size})...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Generate with Zipf-like distribution (more realistic than uniform)
    tokens = rng.zipf(1.5, size=n_tokens).astype(np.int64)
    tokens = tokens % vocab_size
    _check_uint16(tokens, vocab_size)

    # Save as binary
    train_path = output_dir / 'train.bin'
    val_path = output_dir / 'val.bin'

    split = int((1.0 - val_frac) * n_tokens)
    tokens[:split].astype(np.uint16).tofile(train_path)
    tokens[split:].astype(np.uint16).tofile(val_path)

    print(f"Saved: {train_path} ({split:,} tokens, {train_path.stat().st_size/1e6:.1f}MB)")
    print(f"Saved: {val_path} ({n_tokens-split:,} tokens, {val_path.stat().st_size/1e6:.1f}MB)")


def prepare_hf_dataset(dataset_name: str, tokenizer_name: str,
                       output_dir: str, max_tokens: Optional[int] = None, val_frac: float = 0.05):
    """Tokenize a HuggingFace dataset."""
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: Install transformers and datasets:")
        print("  pip install transformers datasets --break-system-packages")
        sys.exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"Loading dataset: {dataset_name}")
    if dataset_name == 'openwebtext':
        ds = load_dataset('openwebtext', split='train', trust_remote_code=True)
    elif dataset_name == 'c4':
        ds = load_dataset('allenai/c4', 'en', split='train',
                          streaming=True, trust_remote_code=True)
    else:
        ds = load_dataset(dataset_name, split='train', trust_remote_code=True)

    print("Tokenizing...")
    all_tokens = []
    total = 0

    for i, example in enumerate(ds):
        text = example.get('text', example.get('content', ''))
        if not text:
            continue

        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        total += len(tokens)

        if i % 10000 == 0:
            print(f"  Processed {i:,} documents, {total:,} tokens...")

        if max_tokens and total >= max_tokens:
            break

    tokens = np.array(all_tokens[:max_tokens or total], dtype=np.int64)
    _check_uint16(tokens, int(tokens.max()) + 1 if tokens.size else 1)
    tokens = tokens.astype(np.uint16)

    # Train/val split
    split = int((1.0 - val_frac) * len(tokens))
    train_path = output_dir / 'train.bin'
    val_path = output_dir / 'val.bin'

    tokens[:split].tofile(train_path)
    tokens[split:].tofile(val_path)

    print(f"\nDone! Total tokens: {len(tokens):,}")
    print(f"  Train: {train_path} ({split:,} tokens)")
    print(f"  Val:   {val_path} ({len(tokens)-split:,} tokens)")


def prepare_local(input_dir: str, tokenizer_name: str,
                  output_dir: str, max_tokens: Optional[int] = None, val_frac: float = 0.05):
    """Tokenize local text files."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: Install transformers:")
        print("  pip install transformers --break-system-packages")
        sys.exit(1)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    all_tokens = []
    files = sorted(input_dir.glob('*.txt'))
    print(f"Found {len(files)} text files in {input_dir}")

    for f in files:
        text = f.read_text(encoding='utf-8', errors='ignore')
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        print(f"  {f.name}: {len(tokens):,} tokens")

        if max_tokens and len(all_tokens) >= max_tokens:
            break

    tokens = np.array(all_tokens[:max_tokens or len(all_tokens)], dtype=np.int64)
    _check_uint16(tokens, int(tokens.max()) + 1 if tokens.size else 1)
    tokens = tokens.astype(np.uint16)

    split = int((1.0 - val_frac) * len(tokens))
    train_path = output_dir / 'train.bin'
    val_path = output_dir / 'val.bin'

    tokens[:split].tofile(train_path)
    tokens[split:].tofile(val_path)

    print(f"\nTotal: {len(tokens):,} tokens")
    print(f"  Train: {train_path} ({split:,})")
    print(f"  Val:   {val_path} ({len(tokens)-split:,})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare training data')
    parser.add_argument('--source', choices=['hf', 'huggingface', 'local', 'synthetic'],
                        default='synthetic')
    parser.add_argument('--dataset', type=str, default='openwebtext',
                        help='HuggingFace dataset name')
    # --input_dir is canonical; --input_path accepted as an alias (README uses it)
    parser.add_argument('--input_dir', '--input_path', type=str, default='./corpus',
                        dest='input_dir',
                        help='Directory of local text files')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        help='HuggingFace tokenizer name')
    # --output is canonical; --output_dir accepted as an alias (README uses it)
    parser.add_argument('--output', '--output_dir', type=str, default='./data',
                        dest='output',
                        help='Output directory')
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='Max tokens to process')
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--val_frac', type=float, default=0.05,
                        help='fraction of the corpus held out as a DISJOINT val.bin')

    args = parser.parse_args()

    if args.source == 'synthetic':
        # Honor --max_tokens when provided; otherwise default corpus size.
        prepare_synthetic(args.output, args.vocab_size, val_frac=args.val_frac,
                          n_tokens=args.max_tokens or 10_000_000)
    elif args.source in ('hf', 'huggingface'):
        prepare_hf_dataset(args.dataset, args.tokenizer, args.output,
                           max_tokens=args.max_tokens, val_frac=args.val_frac)
    elif args.source == 'local':
        prepare_local(args.input_dir, args.tokenizer, args.output,
                      max_tokens=args.max_tokens, val_frac=args.val_frac)
