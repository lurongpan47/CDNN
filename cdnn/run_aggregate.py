#!/usr/bin/env python3
"""Run the MNIST experiment with multiple seeds and report mean +/- std."""
import json
import numpy as np
import sys
sys.path.insert(0, '.')
from cd_mnist_experiment import main as run_experiment

SEEDS = [0, 1, 2]
EPOCHS = 25

results_per_seed = []
for s in SEEDS:
    print(f"\n\n############### SEED {s} ###############")
    out = run_experiment(epochs=EPOCHS, seed=s)
    results_per_seed.append(out)

# Aggregate
print("\n\n" + "=" * 75)
print(f"AGGREGATE OVER {len(SEEDS)} SEEDS (mean +/- std)")
print("=" * 75)
print(f"{'Model':<18}{'#params':>10}{'final loss':>22}"
      f"{'final acc':>20}{'cond #':>15}")
print("-" * 85)
for key, name in [("Dense", "Dense MLP"),
                  ("CD_block4", "CD-MLP (B=4)"),
                  ("CD_block8", "CD-MLP (B=8)")]:
    losses = [r["models"][key]["history"]["train_loss"][-1] for r in results_per_seed]
    accs   = [r["models"][key]["history"]["test_acc"][-1]   for r in results_per_seed]
    conds  = [r["models"][key]["condition_number"]          for r in results_per_seed]
    p      = results_per_seed[0]["models"][key]["params"]
    print(f"{name:<18}{p:>10d}"
          f"  {np.mean(losses):.4f} +/- {np.std(losses):.4f}"
          f"   {np.mean(accs):.4f} +/- {np.std(accs):.4f}"
          f"   {np.mean(conds):.1e}")

# Save aggregate
with open('cd_mnist_aggregate.json', 'w') as f:
    json.dump({
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "by_model": {
            key: {
                "params": results_per_seed[0]["models"][key]["params"],
                "final_loss_mean": float(np.mean([r["models"][key]["history"]["train_loss"][-1] for r in results_per_seed])),
                "final_loss_std":  float(np.std ([r["models"][key]["history"]["train_loss"][-1] for r in results_per_seed])),
                "final_acc_mean":  float(np.mean([r["models"][key]["history"]["test_acc"][-1]   for r in results_per_seed])),
                "final_acc_std":   float(np.std ([r["models"][key]["history"]["test_acc"][-1]   for r in results_per_seed])),
                "cond_mean":       float(np.mean([r["models"][key]["condition_number"]          for r in results_per_seed])),
                "cond_std":        float(np.std ([r["models"][key]["condition_number"]          for r in results_per_seed])),
            } for key in ["Dense", "CD_block4", "CD_block8"]
        }
    }, f, indent=2)
print("\nSaved: cd_mnist_aggregate.json")
