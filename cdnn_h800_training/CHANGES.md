# CD-Transformer — Bug Fixes & Optimization Changelog

Review of the 6-file CD-Transformer codebase for the 8× H800 (single-node, China)
target. Every file was corrected and component-tested on CPU (torch 2.12). The
distributed/FSDP and CUDA paths can't run in this CPU sandbox, so those were
validated by import + isolated component tests; the rest ran end-to-end.

The headline fix is a **CDLinear einsum bug that crashed any run where
batch_size ≠ block_size** — i.e. essentially every real configuration.

---

## Critical bugs (would crash or silently corrupt training)

### 1. `cd_layers.py` — CDLinear einsum label collision *(crash)*
`einsum('oib,bib->bob', ...)` reused the label `b` for **both** the batch axis and
the block-frequency axis. PyTorch requires a matched axis to have equal length, so
the op raised `RuntimeError` the moment `batch != block_size`. Rewritten as
`'oik,nik->nok'` (n = batch/token, k = frequency bin). Verified numerically against
a direct first-column circulant matmul (max error 1.8e-15), confirming the
`ifft(fft(c) * fft(x))` convolution convention was correct — only the labels were wrong.

### 2. `cd_layers.py` — Fisher regularizer contributed zero gradient *(silent no-op)*
`fisher_reg_loss` was computed from `hessian_spectrum()`, which runs under
`torch.no_grad()`. The "regularizer" therefore had no gradient and never affected
training. Re-derived in closed form via Parseval's identity — `Σ_k |FFT(c)|² =
block · ||c||²` — computed directly from the parameter tensor so it is differentiable
and FSDP/autocast-safe. Confirmed gradient now flows to the circulant weights.

### 3. `cd_model.py` — MTP heads attended without a causal mask *(label leakage)*
The Multi-Token Prediction path called attention without `is_causal`, letting each
position peek at future tokens — inflating MTP accuracy and corrupting the auxiliary
signal. Now uses `is_causal=True` consistently with the main trunk.

### 4. `train_distributed.py` — GradScaler used under BF16 *(incorrect / wasteful)*
`torch.cuda.amp.GradScaler` is for FP16 only; under BF16 it is at best a no-op and at
worst destabilizes early steps. Removed the scaler and switched to the current
`torch.amp` API. BF16 needs no loss scaling.

### 5. `train_distributed.py` — FSDP optimizer checkpoint was unresumable *(broken resume)*
Saved a raw `optimizer.state_dict()` under FSDP, which stores shard-local state that
can't be reloaded correctly. Now uses `FSDP.optim_state_dict` /
`optim_state_dict_to_load` within a `state_dict_type` context, and `--resume` is
actually implemented (model + optimizer + scheduler + step).

### 6. `train_distributed.py` — whole corpus loaded into RAM *(OOM on real data)*
`TokenDataset` did `np.memmap(...).astype(np.int64)`, which materializes the entire
token file in memory (and doubles it 16→64-bit). Switched to a lazy `uint16` memmap
with on-the-fly next-token shift and `ignore_index=-100` padding.

---

## Correctness & performance fixes

### 7. `cd_layers.py` — CDMoELayer routing rewrite *(perf + balancing)*
- Replaced the `O(n_active × n_experts)` nested loop (with a per-iteration `.any()`
  host-sync that serializes the GPU) with an `O(n_experts)` grouped dispatch using
  `index_add_`.
- The learnable load-balancing **bias was never updated**, so the
  "auxiliary-loss-free balancing" did nothing — now updated once per optimizer step.
- Gate combine-weights were taken from the **bias-adjusted** logits; the bias is meant
  to steer *selection* only. Weights now come from the raw logits; bias affects top-k
  choice alone (DeepSeek-V3 behavior).

### 8. `cd_layers.py` — CDAttention uses fused SDPA *(perf)*
Replaced the manual `softmax(QKᵀ/√d + mask)` with
`F.scaled_dot_product_attention(..., is_causal=True)` so it dispatches to
FlashAttention/efficient kernels on H800. Removed a dead `kv_rope` split.

### 9. `cd_layers.py` — gradient-checkpoint-safe MoE bias *(subtle correctness)*
The first version of the bias-update fix mutated `expert_bias` *inside* `forward`,
which breaks gradient checkpointing: the recompute pass saw a different bias →
different routing → tensor-shape mismatch. `forward` is now side-effect-free; per-step
load is stashed via an idempotent `copy_` into a `last_load` buffer, and a separate
`update_router_biases(model)` applies the update **outside** the checkpointed region,
once per optimizer step.

### 10. `cd_layers.py` / `cd_model.py` — misc
- `fp8_cast` returned a tuple in one branch and a bare tensor in another → unpack
  crash; made consistent.
- `create_model` mutated the shared `CONFIGS` template in place (config bleed across
  calls) → switched to `dataclasses.replace`.
- Fixed a reversed `isinstance` check in a self-test.
- Fused AdamW now guarded behind a CUDA check; epoch-loss averaging corrected;
  FSDP-aware `model.clip_grad_norm_` used for gradient clipping.

### 11. `prepare_data.py` — uint16 overflow guard + arg/synthetic fixes
- Token ids are stored as `uint16`; a vocab > 65535 (DeepSeek/Llama scale) silently
  wraps and corrupts the corpus. Added a loud guard on both vocab size and observed
  max id.
- `--max_tokens` is now honored for `--source synthetic` (previously ignored).
- Accepts the arg spellings the README documents (`--output_dir`, `--input_path`,
  `--source huggingface`) as aliases.

### 12. `launch_h800.sh` — safer single-node NCCL defaults
The script forced `NCCL_IB_DISABLE=0` and pinned `NCCL_SOCKET_IFNAME=eth0`. On a
single-node box all 8 GPUs use NVLink (no InfiniBand), and many China-hosted H800
servers have no RDMA NIC or a differently-named interface, so the original settings
can hang NCCL at init. Now disables IB for the single-node run, lets NCCL auto-detect
the interface, keeps `NCCL_P2P_LEVEL=NVL`, and documents how to re-enable IB/GDR for
multi-node.

### 13. `README.md` — doc/flag alignment + honesty
- Aligned `--resume` (was `--resume_from`) and the prepare-data flags with the code.
- Added an explicit **FP8 status note** (see below).
- Marked the 310× κ / 97.50% MNIST numbers as the paper's MNIST figures, not
  re-validated at transformer scale in this repo.

---

## About FP8 (please read)

The `--use_fp8` flag, `fp8_cast`, and `fp8_matmul` are **scaffolding only** — the
quantize helper is not on the hot compute path, so the actual GEMMs run in **BF16**.
Enabling the flag does not currently change numerics or throughput. I kept the helpers
as honest, clearly-labeled stubs rather than leaving a broken cast on the compute path.

For real FP8 throughput on H800 (Hopper), integrate DeepSeek's open-source kernels
instead of a hand-rolled cast — all MIT-licensed:

- **DeepGEMM** — FP8 dense + grouped/masked MoE GEMMs, JIT, ~1350+ TFLOPS on Hopper.
  Use `m_grouped_gemm_fp8_fp8_bf16_nt_contiguous` for the expert path.
- **FlashMLA** — Hopper MLA decode kernel (BF16, paged KV, block 64) for attention
  during decode.
- **DeepEP** — expert-parallel all-to-all dispatch/combine (FP8 dispatch over NVLink)
  if MoE scales past one node.

Until then, BF16 AMP (`--use_amp`) is the correct, stable default and is what the loop
actually uses.

---

## Testing performed

- `cd_layers.py` — all layer smoke tests pass including backward; einsum verified
  against direct circulant matmul (err 1.8e-15); Fisher gradient flow confirmed; MoE
  balancer exercised.
- `cd_model.py` — tiny-config forward/backward verified including gradient
  checkpointing + MTP + Fisher. (A 403M-param backward was OOM-killed by the CPU
  sandbox's memory limit — an environment limit, not a code bug.)
- `train_distributed.py` — imports clean; `TokenDataset` (lazy memmap, next-token
  shift, padding) and the cosine-warmup scheduler verified. FSDP/NCCL path validated
  by import + component tests only (no CUDA/NCCL in this sandbox).
- `prepare_data.py` — synthetic generation, `--max_tokens`, and the uint16 overflow
  guard all verified end-to-end.
- `launch_h800.sh` — `bash -n` syntax-clean; arg names cross-checked against the
  training script's argparse.

Recommend a first real-hardware run with `./launch_h800.sh small` on synthetic data to
confirm NCCL init and FSDP wrapping before committing to a long job.

## Reconciliation + baseline + scaling fixes

(a) CE-logging discrepancy
- cd_model.py: forward now exposes `result['ce_loss']` = the bare next-token cross-entropy.
- train_distributed.py: logs `CE:` and `PPL:` DIRECTLY (averaged over the grad-accum
  window), plus correctly-averaged MTP/Fisher — no more decoding the total loss.
- reconcile_ce.py: diagnostic that, on one batch, separates (1) decode error,
  (2) train-vs-eval forward gap (Shannon dropout / MoE path), and (3) data identity
  (checksum of train.bin vs val.bin) to localize the ~1.2-nat gap.

(b) Dense baseline
- cd_layers.py: CDLinear `impl="dense"` = full non-circulant weight, same module
  interface; Fisher/conditioning skip dense layers (c is None). impl threaded through
  CDAttention / CDMoELayer / CDTransformerBlock.
- cd_model.py: `cd_impl` config field. train_distributed.py: `--dense` (and `--cd_impl`).
  Run the baseline with identical architecture & schedule: `DENSE=1 ./launch_h800.sh small`.

(c) Properly-resourced run
- cd_layers.py / cd_model.py: conditioning aggregation `--fisher_agg {mean,pnorm,max}`
  (+ `--fisher_p`) so worst blocks aren't diluted (mean<pnorm<max emphasis verified).
- train_distributed.py: `--warmup_frac` sets warmup as a fraction of total steps
  (fixes the warmup>total LR-cap bug across dataset sizes).
- prepare_data.py: `--val_frac` for the size of the DISJOINT held-out val.bin.

## Report-driven performance optimizations

cd_layers.py — CDMoELayer dispatch (the "immediate engineering follow-on" the
  report/paper flagged as the main residual throughput bottleneck):
  - was: a Python loop doing E full boolean scans `(flat_expert == e)` (each
    O(N*k)) plus a tensor allocation per expert.
  - now: argsort the (token,slot) assignments by expert ONCE, process each expert
    as a contiguous slice, single host sync for per-expert counts. Verified
    numerically identical to the old loop (forward 6e-8, grad 1.9e-6).

train_distributed.py — HessianMonitor (the source of the logged-step throughput
  dip in profiling):
  - the full-model summon_full_params all-gather now runs on its own cadence
    (`--hessian_interval`, default 500) instead of every loss-log step (~10x fewer
    expensive collectives), inspects at most `max_layers` CDLinear layers for a
    representative kappa, and correctly skips dense-baseline layers (c is None,
    which previously would have crashed).
