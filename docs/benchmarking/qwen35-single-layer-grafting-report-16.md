# Qwen3.5 single-layer graft Report 16

## Summary

Report 16 was the top arm in the 30-arm, 1M-token single-layer graft screen. It
inserted frozen Qwen3.5-27B donor layer 32 into the complete frozen Qwen3.5-9B
host using a ridge-initialized bounded residual stitch.

The arm remained close to the unmodified 9B baseline through 25M continuation
tokens, but it did not improve with additional training. Its combined
MMLU-Pro and BBH score declined from `1.134236` at 1M to `1.131933` at 25M, a
change of `-0.002303`.

## Configuration

- Host: complete Qwen3.5-9B, frozen
- Donor: Qwen3.5-27B layer 32, frozen and always executed
- Insertion point: after host layer 15
- Operation: frozen-stitch bounded RMS residual
- Initialization: layer-32 ridge projection
- Maximum residual scale: `0.05`
- Retention KL weight: `1.0`
- Projection anchor weight: `0.0001`
- Scheduler horizon: 100M tokens
- Protocol identity:
  `b82fc1daf569d506e0a1ff24d1e96674879315248311badd10966164a685e111`

Evaluation covered 12,032 MMLU-Pro samples and 5,761 BBH samples.

## Results

| Checkpoint | MMLU-Pro | BBH | Combined | Change from 1M |
|---:|---:|---:|---:|---:|
| 1M | 0.514378 | 0.619858 | 1.134236 | -- |
| 25M | 0.513464 | 0.618469 | 1.131933 | -0.002303 |

From 1M to 25M, MMLU-Pro changed by `-0.000914` and BBH changed by
`-0.001389`. The change is small, but both tasks moved in the same direction.

## Interpretation

The result validates the promotion runtime and shows that this bounded frozen
donor graft can retain approximately baseline-level quality over a longer
continuation. It does not show a capability gain from training this arm beyond
the 1M pilot. Selection of the 50M cohort should compare all twelve promoted
arms at the same 25M boundary rather than advancing Report 16 on its 1M rank.

## Artifact evidence

The local campaign artifacts are not committed to this repository. These
SHA-256 hashes bind the report to immutable completion receipts.

| Checkpoint | Training completion SHA-256 | Evaluation completion SHA-256 |
|---:|---|---|
| 1M | `b7a2e8a47c5cad8fca63331e8ea2fb344bf16c5088630317e984cde1e1e2737c` | `b856b06cd6c2aa62ce82c1f416e5b7e9099867ca82751995b4db77e9c7c4a89d` |
| 25M | `cf47400f5d32d88d4cb1f1b4acd4c2476356afd441d12292f1f401033bde0c22` | `e9fb3cd96b1167379b1063362131765e915b8f81075c1750042218bbe6df2066` |

The 25M checkpoint binds runtime SHA-256
`f79872fd43ee3e62be0056e6832a3bb54b80f2bb7ebcc485b988e7f236b3e1c1`
and core SHA-256
`1881fbc8289613345ce965d1b46fc47a88b64fe934991a470897f0a5403e86a2`.
