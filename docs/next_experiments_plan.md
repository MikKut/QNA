# Next experiments plan: QNA scouting v1

## Goal

Find whether there exists a low-shot regime in the current architecture where **QNA-Adam**
outperforms **Adam** in validation accuracy, and verify that any improvement is **not**
explained only by a smaller global learning rate.

## Context from completed experiments

Already tested:
- train shots: 100, 36, 15
- same base hyperparameters:
  - lr = 3e-4
  - betas = (0.9, 0.999)
  - eps = 1e-8
  - weight_decay = 0
  - eval_shots = 2048
  - reseed_each_epoch = false
  - same batch size / epochs

Observed:
- QNA and Adam are very close at shots = 100 and 36
- at shots = 15 both degrade strongly, but QNA does not yet show a clear accuracy win
- QNA scaling is active, especially at lower shots, but this has not yet translated into a clear validation-accuracy advantage

## Principle for the next step

Do **not** change architecture yet.

First, perform a **scouting package** in a more noise-dominated regime while keeping the
current architecture fixed.

This isolates the optimizer effect.

## Scouting package (single seed)

Use:
- seed = 42
- train shots in {7, 15, 25}
- optimizers:
  - Adam
  - Adam-matched
  - QNA-Adam
- lambda_var for QNA in {8, 16, 32}
- epochs = 20
- eval_shots = 2048
- keep all other settings unchanged

## Why these settings

### Shots
- 7: stronger noise regime, best chance for QNA to show benefit
- 15: connects to previous experiments
- 25: bridge between 15 and 36

### Adam-matched
Purpose:
- control experiment to test whether QNA wins only because its effective step is smaller

Definition:
- Adam uses lr = 3.0e-4
- Adam-matched uses lr = 2.7e-4
- QNA uses lr = 3.0e-4

This is a simple first-order control, roughly corresponding to a modest global reduction
in effective step size.

### QNA lambdas
Use logarithmic spacing:
- 8
- 16
- 32

This is enough for scouting without exploding the experiment count.

## Total number of runs

Single seed package:
- Adam: 3 runs
- Adam-matched: 3 runs
- QNA: 3 shots × 3 lambdas = 9 runs

Total = 15 runs

## Expected success pattern

Ideal outcome:

- at shots = 7:
  - QNA val_acc > Adam
  - QNA val_acc > Adam-matched

- at shots = 15:
  - QNA is at least competitive and preferably slightly better

- at shots = 25:
  - difference becomes smaller

This would support the claim that QNA is useful specifically in the low-shot regime.

## What to compare after the runs

For each config, record:
- optimizer
- shots train
- lambda_var
- best epoch
- best val_loss
- best val_acc
- final / best train_loss
- final / best train_acc
- grad norm
- grad norm raw
- scale_mean
- scale_min
- update norm
- cos_gu

## Decision after scouting

### If a clear QNA signal appears
Take the best regime, for example:
- shots = 7
- lambda = 16

Then run a second-stage validation:
- seeds = {42, 43, 44}
- Adam
- Adam-matched
- QNA(best lambda)

This checks that the effect is not a lucky single-seed result.

### If no clear signal appears
Only then consider architecture changes, in this order:
1. increase training difficulty slightly (more classes or deeper ansatz)
2. reduce shots even further
3. simplify/adjust measurement choice
4. revisit QNA variance normalization and lambda schedule

## Important note

At this stage, do not change:
- PCA dimension
- n_layers
- measurement scheme
- dataset split
- eval_shots

The current question is:
**does QNA help in the current architecture when train-shot noise becomes stronger?**