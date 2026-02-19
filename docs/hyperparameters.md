# Hyperparameter Notes

This page documents the non-obvious configuration keys and recommended defaults.
For most runs, start from `config/ensemble_3models.yaml` and only override a small subset.

## Core Defaults (Recommended)

1. `optim.align=pooler_weighted`
2. `attack=pgd_multi_pass`
3. `blackbox.model_name=["gpt5-thinking-low","claude4.0","gemini2.5pro"]`

## Important Clarifications

## 1) `optim.align`

1. Purpose: alignment method used by the attack objective.
2. Recommendation: keep `pooler_weighted`.
3. Status: this is the best-performing setting used throughout experiments.

## 2) `optim.tm_idx` (Deprecated)

1. Original purpose: layer indices for intermediate-layer matching.
2. Current status: kept only for compatibility with older experiments.
3. Recommendation: do not tune this for current release runs.

## 3) `optim.beta`

1. Meaning: this corresponds to `lambda` in the paper.
2. Applies to: `pooler_weighted` alignment.

## 4) `optim.multi_pass_num`

1. Meaning: this corresponds to `K` in the paper.
2. Effect: controls multi-pass count in `pgd_multi_pass`.

## 5) `model.target_num`

1. Meaning: this is `p + 1`.
2. Why `+1`: one fixed target image is always included; remaining targets come from retrieval.

## 6) `blackbox.model_name`

Recommended trio for release evaluation:

1. `gpt5-thinking-low`
2. `claude4.0`
3. `gemini2.5pro`

Supported names and aliases:

| Family | Recommended Name | Accepted Aliases | Backend Model Used | API Key Bucket in `api_keys.yaml` |
|---|---|---|---|---|
| GPT-5 | `gpt5-thinking-low` | any string starting with `gpt5` or `gpt-5` (e.g., `gpt5`, `gpt-5`, `gpt5-thinking-medium`, `gpt5-thinking-high`) | `gpt-5-2025-08-07` | `gpt4o` |
| GPT-4o | `gpt4o` | `gpt4o` | `gpt-4o` | `gpt4o` |
| O3 | `o3` | `o3` | `o3` | `gpt4o` |
| Claude 4.0 | `claude4.0` | `claude4`, `claude-4`, `claude-4.0` | `claude-sonnet-4-20250514` | `claude` |
| Claude 4.0 Thinking | `claude4.0t` | `claude4t`, `claude-4t`, `claude-4.0t` | `claude-sonnet-4-20250514` (thinking enabled) | `claude` |
| Claude 3.7 | `claude3.7` | `claude3.7` | `claude-3-7-sonnet-20250219` | `claude` |
| Claude 3.7 Thinking | `claude3.7t` | `claude3.7t` | `claude-3-7-sonnet-20250219` (thinking enabled) | `claude` |
| Claude Base | `claude` | `claude` | `claude-3-5-sonnet-20241022` | `claude` |
| Gemini 2.5 Pro | `gemini2.5pro` | `gemini2.5` | `gemini-2.5-pro-preview-03-25` | `gemini` |
| Gemini 2.5 Flash | `gemini2.5flash` | `gemini2.5flash` | `gemini-2.5-flash-preview-04-17` | `gemini` |
| Gemini Base | `gemini` | `gemini` | `gemini-2.0-flash` | `gemini` |

Note:

1. For GPT-5 aliases containing `thinking-*`, the suffix controls reasoning effort (`low`, `medium`, `high`, `minimal`).
2. In this codebase, GPT-family models (`gpt*` and `o3`) read keys from the `gpt4o` entry in `api_keys.yaml`.

## 7) `attack`

1. `pgd_multi_pass`: main M-Attack-V2 attack (recommended default).
2. `fgsm`, `mifgsm`, `pgd`: lightweight baselines.

## 8) `generated_img_hash`

1. Typical use: `null` during end-to-end pipeline runs.
2. Advanced use: set this manually during evaluation-only runs when reusing existing generated images and you want to override hash resolution.
3. Practical scenario: generate once, evaluate multiple times, but you no longer remember the exact generation config.

## W&B Fields

1. `wandb.entity`: set to your own account/entity (`"???"` placeholder in config).
2. `wandb.project`: use your own project name template.

## Common Safe Overrides

```bash
uv run python generate_ad_sample_parallel.py \
  data.cle_data_path=resources/images/bigscale_100 \
  data.tgt_data_path=resources/images/target_images_100 \
  data.num_samples=100 \
  optim.alpha=0.005 \
  optim.epsilon=16 \
  attack=pgd_multi_pass
```
