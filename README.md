# Convergence Analysis of SGD under Expected Smoothness
Source code for reproducing our paper's experiments.

## Abstract
Stochastic gradient descent (SGD) is the workhorse of large-scale learning, yet classical analyses rely on assumptions that can be either too strong (bounded variance) or too coarse (uniform noise). The expected smoothness (ES) condition has emerged as a flexible alternative that ties the second moment of stochastic gradients to the objective value and the full gradient. This paper presents a self-contained convergence analysis of SGD under ES. We (i) refine ES with interpretations and sampling-dependent constants; (ii) derive bounds of the expectation of squared full gradient norm; and (iii) prove O(1/K) rates with explicit residual errors for various step-size schedules. All proofs are given in full detail in the appendix. Our treatment unifies and extends recent threads (Khaled and Richtárik, 2020; Umeda and Iiduka, 2025).

## Wandb Setup

```bash
wandb.init(config=args, project=wandb_project_name, name=wandb_exp_name)
```

## Usage
To train the model, please replace `XXXXXX` with your desired sampling method (`normal`, `replacement`, `independent`, or `tau-nice`).

```bash
# Basic usage
python main.py

# Usage with WandB and specific sampling
nohup python main.py --use_wandb --sampling XXXXXX > log.log &
```

## Example JSON Configuration
The following is an example configuration for training ResNet18 using the SGD optimizer.   Batch size, learning rate, and momentum are all set to constant values, and training runs for 300 epochs.

```json
{
  "optimizer": "sgd",
  "model": "resnet18",
  "bs_method": "constant",
  "lr_method": "constant",
  "init_bs": 128,
  "init_lr": 0.1,
  "epochs": 300
  "use_wandb": true
}
```

Below is a detailed description of each configuration parameter available in `main.py`:

| Parameter | Type & Example | Description |
| :- | :- | :- |
| `--optimizer` | `str` (`"sgd"`, `"momentum"`, `"adam"`,<br>`"rmsprop"`, `"adagrad"`, `"adamw"`, `"amsgrad"`) | Specifies the optimizer to use during training. Default is `"sgd"`. |
| `--sampling` | `str` (`"normal"`, `"replacement"`,<br>`"independent"`, `"tau-nice"`) | Specifies the sampling strategy. This controls the weighting vector $v$ applied to the loss function. Default is `"normal"`. |
| `--batch_size` | `int` (`128`) | Training batch size. Also used to calculate sampling probabilities $p$ and $q$. Default is `128`. |
| `--use_wandb` | `flag` (no value required) | If present, enables logging to Weights & Biases (wandb). Logs include ES metrics (LHS, RHS), gradient norms, and training accuracy. |
```
