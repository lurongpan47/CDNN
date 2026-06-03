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

# ---------------------------------------------------------------------------
# China access: default HuggingFace traffic to the hf-mirror.com endpoint.
# This MUST be set before `transformers`/`huggingface_hub` are imported, so we
# do it at module load. Override by exporting HF_ENDPOINT yourself, or set it
# to the empty string to force the real huggingface.co.
# ---------------------------------------------------------------------------
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')


# ---------------------------------------------------------------------------
# Robust tokenizer loading (works offline / on air-gapped GPU nodes)
# ---------------------------------------------------------------------------
class _TiktokenWrapper:
    """Adapter so a tiktoken encoding exposes a .encode() like HF tokenizers."""
    def __init__(self, enc, name):
        self._enc = enc
        self.name = name
        # GPT-2 BPE vocab is 50257; expose for sanity checks
        self.vocab_size = enc.n_vocab

    def encode(self, text):
        # disallow special tokens to avoid surprises on raw corpora
        return self._enc.encode(text, disallowed_special=())


class _ByteTokenizer:
    """Zero-dependency, zero-network byte-level tokenizer.

    Encodes text as raw UTF-8 bytes → vocab of 256. Always available,
    never touches the network. Useful for air-gapped nodes or quick runs.
    Note: byte-level means longer sequences than BPE, but it trains fine
    and requires the model's vocab_size to be >= 256 (it is: default 32000).
    """
    name = 'byte'
    vocab_size = 256

    def encode(self, text):
        return list(text.encode('utf-8', errors='ignore'))


def load_tokenizer(tokenizer_name: str):
    """Load a tokenizer, preferring sources reachable from inside China.

    Resolution order:
      1. 'byte' → built-in byte-level tokenizer (truly offline, no deps)
      2. tiktoken (bundled BPE) for gpt2 / gpt-4 names
      3. ModelScope (魔搭, reachable in China) — used automatically for
         'deepseek'/'qwen' shortcuts, or any id when --use_modelscope is set
      4. HuggingFace AutoTokenizer (honors HF_ENDPOINT mirror + offline cache)
      5. Clear, actionable error message

    DeepSeek shortcuts (resolve to ModelScope ids, downloadable in China):
      'deepseek'      → deepseek-ai/DeepSeek-V2 (vocab ~100k, BPE)
      'deepseek-v3'   → deepseek-ai/DeepSeek-V3
      'deepseek-coder'→ deepseek-ai/deepseek-coder-6.7b-base
    """
    name = tokenizer_name.lower().strip()

    # --- 1. byte-level: always works, no network, no deps ---
    if name in ('byte', 'bytes', 'utf8', 'utf-8'):
        print("Loaded built-in byte-level tokenizer (vocab=256, fully offline)")
        return _ByteTokenizer()

    # --- 2. tiktoken: bundled BPE (vocab cached locally after first use) ---
    tiktoken_map = {
        'gpt2': 'gpt2',
        'gpt-2': 'gpt2',
        'r50k_base': 'r50k_base',
        'p50k_base': 'p50k_base',
        'cl100k_base': 'cl100k_base',   # GPT-3.5 / GPT-4
        'gpt-4': 'cl100k_base',
        'gpt-3.5': 'cl100k_base',
        'o200k_base': 'o200k_base',     # GPT-4o
        'gpt-4o': 'o200k_base',
    }
    if name in tiktoken_map:
        try:
            import tiktoken
            enc = tiktoken.get_encoding(tiktoken_map[name])
            print(f"Loaded tokenizer via tiktoken: {tiktoken_map[name]} "
                  f"(vocab={enc.n_vocab})")
            return _TiktokenWrapper(enc, tiktoken_map[name])
        except ImportError:
            print("  (tiktoken not installed; trying model hubs)")
        except Exception as e:
            print(f"  (tiktoken unavailable offline: {e})")

    # --- DeepSeek / Qwen shortcuts → ModelScope ids (China-reachable) ---
    deepseek_map = {
        'deepseek': 'deepseek-ai/DeepSeek-V2',
        'deepseek-v2': 'deepseek-ai/DeepSeek-V2',
        'deepseek-v3': 'deepseek-ai/DeepSeek-V3',
        'deepseek-coder': 'deepseek-ai/deepseek-coder-6.7b-base',
        'qwen': 'qwen/Qwen2.5-7B',
        'qwen2.5': 'qwen/Qwen2.5-7B',
    }

    # Explicit "modelscope:<id>" form, e.g. --tokenizer modelscope:deepseek-ai/DeepSeek-V2
    if name.startswith('modelscope:'):
        ms_id = tokenizer_name.split(':', 1)[1]
        tok = _try_modelscope(ms_id)
        if tok is not None:
            return tok
        print(f"ERROR: ModelScope could not load '{ms_id}'.")
        _print_china_tokenizer_help(ms_id)
        sys.exit(1)

    # Shortcut names are ModelScope-ONLY: do not fall through to huggingface.co
    # (that produced the misleading 'Can't load deepseek' error before).
    if name in deepseek_map:
        resolved_id = deepseek_map[name]
        print(f"Resolving '{tokenizer_name}' → ModelScope id '{resolved_id}'")
        tok = _try_modelscope(resolved_id)
        if tok is not None:
            return tok
        print(f"ERROR: Could not load DeepSeek/Qwen tokenizer '{resolved_id}' "
              f"from ModelScope.")
        _print_china_tokenizer_help(resolved_id)
        sys.exit(1)

    resolved_id = tokenizer_name
    prefer_modelscope = False

    # --- 4. HuggingFace AutoTokenizer (respects HF_ENDPOINT mirror + cache) ---
    # HF_ENDPOINT was defaulted to https://hf-mirror.com at module load so that
    # huggingface traffic resolves from inside China. Override via env if needed.
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(f"ERROR: No usable tokenizer backend for '{tokenizer_name}'.")
        print("  Easiest fix (no network, no deps):  --tokenizer byte")
        print("  Or install a backend:")
        print("    pip install tiktoken          # GPT-2/4 BPE")
        print("    pip install modelscope        # DeepSeek/Qwen, China-friendly")
        print("    pip install transformers      # HuggingFace tokenizers")
        sys.exit(1)

    # Guard against a local folder shadowing the model id
    if os.path.isdir(resolved_id) and not os.path.isfile(
        os.path.join(resolved_id, 'config.json')
    ):
        print(f"ERROR: A local directory named '{resolved_id}' exists but has "
              f"no config.json, so it shadows the model id and breaks loading.")
        print(f"  Fix: rename/remove that directory, pass a different "
              f"--tokenizer, or use --tokenizer byte.")
        sys.exit(1)

    try:
        tok = AutoTokenizer.from_pretrained(resolved_id, trust_remote_code=True)
        print(f"Loaded tokenizer via HuggingFace ({os.environ.get('HF_ENDPOINT')}): "
              f"{resolved_id}")
        return tok
    except Exception as e:
        # Last resort: try ModelScope even if we didn't prefer it
        if not prefer_modelscope:
            tok = _try_modelscope(resolved_id)
            if tok is not None:
                return tok
        print(f"ERROR: Could not load tokenizer '{resolved_id}'.")
        print(f"  Underlying error: {e}")
        print("\n  You appear to be offline or behind a network restriction.")
        print("  Options (best first for users in China):")
        print("   1. Fully offline, no deps:  --tokenizer byte")
        print("   2. DeepSeek via ModelScope:  pip install modelscope")
        print("        then  --tokenizer deepseek")
        print("   3. HuggingFace mirror:  export HF_ENDPOINT=https://hf-mirror.com")
        print("        then  --tokenizer deepseek-ai/DeepSeek-V2")
        print("   4. Point --tokenizer at a local dir with the tokenizer files.")
        sys.exit(1)


def _try_modelscope(model_id: str):
    """Attempt to load a tokenizer from ModelScope (魔搭). Returns None on fail.

    ModelScope mirrors many models (DeepSeek, Qwen, etc.) on servers that are
    reachable from inside mainland China without a VPN.
    """
    try:
        from modelscope import AutoTokenizer as MSAutoTokenizer
    except ImportError:
        print("  (modelscope not installed: pip install modelscope)")
        return None
    try:
        tok = MSAutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(f"Loaded tokenizer via ModelScope (魔搭): {model_id} "
              f"(vocab={getattr(tok, 'vocab_size', 'n/a')})")
        return tok
    except Exception as e:
        print(f"  (ModelScope load failed for {model_id}: {e})")
        return None


def _print_china_tokenizer_help(model_id: str):
    """Actionable guidance when a DeepSeek/Qwen tokenizer can't be fetched."""
    print("\n  How to fix (from inside China):")
    print("   1. Install ModelScope and log in (some repos need a free login):")
    print("        pip install modelscope")
    print("        modelscope login        # paste token from modelscope.cn")
    print(f"      then retry:  --tokenizer {model_id}")
    print("   2. Download the tokenizer ONCE, then point --tokenizer at the dir:")
    print("        # on any machine with ModelScope access:")
    print("        from modelscope import snapshot_download")
    print(f"        snapshot_download('{model_id}')   # caches under ~/.cache/modelscope")
    print("        # copy that folder to the GPU node, then:")
    print(f"        python prepare_data.py --source local --input_dir ./corpus \\")
    print(f"            --tokenizer /path/to/{model_id.split('/')[-1]} --output ./data")
    print("   3. No-network fallback that always works:  --tokenizer byte")
    print("\n  NOTE: a DeepSeek tokenizer has ~100k vocab. After prep, set the")
    print("  model's vocab_size to the value printed above (also in meta.json),")
    print("  or training will crash with an index-out-of-range.")


def prepare_synthetic(output_dir: str, vocab_size: int = 32000,
                      n_tokens: int = 10_000_000):
    """Generate synthetic token data for testing."""
    print(f"Generating {n_tokens:,} synthetic tokens (vocab={vocab_size})...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Generate with Zipf-like distribution (more realistic than uniform)
    tokens = rng.zipf(1.5, size=n_tokens).astype(np.int64)
    tokens = tokens % vocab_size

    # Save as binary
    train_path = output_dir / 'train.bin'
    val_path = output_dir / 'val.bin'

    split = int(0.95 * n_tokens)
    tokens[:split].astype(np.uint16).tofile(train_path)
    tokens[split:].astype(np.uint16).tofile(val_path)

    print(f"Saved: {train_path} ({split:,} tokens, {train_path.stat().st_size/1e6:.1f}MB)")
    print(f"Saved: {val_path} ({n_tokens-split:,} tokens, {val_path.stat().st_size/1e6:.1f}MB)")


def prepare_hf_dataset(dataset_name: str, tokenizer_name: str,
                       output_dir: str, max_tokens: Optional[int] = None):
    """Tokenize a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install datasets:")
        print("  pip install datasets --break-system-packages")
        sys.exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = load_tokenizer(tokenizer_name)

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

    all_tokens = all_tokens[:max_tokens or total]
    print(f"\nDone! Total tokens: {len(all_tokens):,}")
    _save_tokens(all_tokens, tokenizer, output_dir)


def prepare_local(input_dir: str, tokenizer_name: str,
                  output_dir: str, max_tokens: Optional[int] = None):
    """Tokenize local text files."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(tokenizer_name)

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

    all_tokens = all_tokens[:max_tokens or len(all_tokens)]
    _save_tokens(all_tokens, tokenizer, output_dir)


def _infer_vocab_size(tokenizer, tokens) -> int:
    """Best-effort tokenizer vocab size, falling back to max observed id + 1."""
    for attr in ('vocab_size', 'n_vocab'):
        v = getattr(tokenizer, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return (int(max(tokens)) + 1) if len(tokens) else 0


def _save_tokens(all_tokens, tokenizer, output_dir: Path):
    """Save tokens to train/val .bin, choosing uint16 or uint32 safely.

    IMPORTANT: uint16 only holds ids 0..65535. DeepSeek/Qwen tokenizers have
    ~100k+ vocab, which would silently overflow uint16 and corrupt the data.
    We pick the dtype from the tokenizer's vocab size and record it.
    """
    vocab = _infer_vocab_size(tokenizer, all_tokens)
    if vocab > 65536:
        dtype = np.uint32
    else:
        dtype = np.uint16
    print(f"\nTokenizer vocab size: {vocab:,} → storing tokens as {dtype.__name__}")

    # Warn appropriately about model vocab_size matching.
    MODEL_DEFAULT_VOCAB = 32000
    if vocab > MODEL_DEFAULT_VOCAB:
        print("  " + "!" * 64)
        print(f"  ACTION REQUIRED: tokenizer vocab ({vocab:,}) EXCEEDS the model's")
        print(f"  default vocab_size ({MODEL_DEFAULT_VOCAB:,}). Training WILL CRASH")
        print(f"  with an index-out-of-range in the embedding/output layer unless")
        print(f"  you set the model vocab_size to {vocab:,}.")
        print(f"  In cd_model_a100.py set the chosen config's vocab_size={vocab},")
        print(f"  or override it when constructing the model.")
        print(f"  Tokens stored as {dtype.__name__}; TokenDataset reads dtype from")
        print(f"  meta.json automatically.")
        print("  " + "!" * 64)
    elif vocab and vocab < MODEL_DEFAULT_VOCAB:
        print(f"  (Info: tokenizer vocab {vocab:,} < model default "
              f"{MODEL_DEFAULT_VOCAB:,}. Training works as-is; for efficiency you")
        print(f"   may set the model vocab_size={vocab} to shrink the embedding.)")

    tokens = np.array(all_tokens, dtype=dtype)
    split = int(0.95 * len(tokens))
    train_path = output_dir / 'train.bin'
    val_path = output_dir / 'val.bin'
    tokens[:split].tofile(train_path)
    tokens[split:].tofile(val_path)

    # Record metadata so training reads the .bin with the correct dtype
    import json
    meta = {
        'vocab_size': int(vocab),
        'dtype': dtype.__name__,
        'tokenizer': getattr(tokenizer, 'name', tokenizer.__class__.__name__),
        'n_train': int(split),
        'n_val': int(len(tokens) - split),
    }
    with open(output_dir / 'meta.json', 'w') as fh:
        json.dump(meta, fh, indent=2)

    print(f"  Train: {train_path} ({split:,} tokens)")
    print(f"  Val:   {val_path} ({len(tokens)-split:,} tokens)")
    print(f"  Meta:  {output_dir/'meta.json'}  (dtype={dtype.__name__})")
    return dtype


if __name__ == '__main__':
    print("prepare_data.py  [v3: byte + tiktoken + ModelScope/DeepSeek + HF-mirror]")
    parser = argparse.ArgumentParser(description='Prepare training data')
    parser.add_argument('--source',
                        choices=['hf', 'huggingface', 'local', 'synthetic'],
                        default='synthetic',
                        help="'hf'/'huggingface', 'local', or 'synthetic'")
    parser.add_argument('--dataset', type=str, default='openwebtext',
                        help='HuggingFace dataset name')
    parser.add_argument('--input_dir', '--input_path', dest='input_dir',
                        type=str, default='./corpus',
                        help='Directory of local .txt files')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        help="Tokenizer: 'byte' (offline, no deps), 'gpt2'/"
                             "'gpt-4' (tiktoken), a HuggingFace id, or a local "
                             "tokenizer dir")
    parser.add_argument('--output', '--output_dir', dest='output',
                        type=str, default='./data',
                        help='Output directory')
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='Max tokens to process')
    parser.add_argument('--vocab_size', type=int, default=32000)

    args = parser.parse_args()

    if args.source == 'synthetic':
        prepare_synthetic(args.output, args.vocab_size)
    elif args.source in ('hf', 'huggingface'):
        prepare_hf_dataset(args.dataset, args.tokenizer, args.output,
                           args.max_tokens)
    elif args.source == 'local':
        prepare_local(args.input_dir, args.tokenizer, args.output,
                      args.max_tokens)
