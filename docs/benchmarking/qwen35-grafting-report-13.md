# Qwen3.5 graft Report 13: causal cross-attention fusion

## Summary

Report 13 evaluated a causal cross-attention graft from Qwen3.5-27B into the
complete Qwen3.5-9B host. The graft improved the combined MMLU-Pro and BBH score
at 100M continuation tokens, but the improvement did not persist through the
250M checkpoint.

The best checkpoint was 100M:

- MMLU-Pro: `0.514628` (`+0.000582` versus the 9B baseline)
- BBH: `0.623503` (`+0.005207` versus the 9B baseline)
- Combined: `1.138131` (`+0.005789` versus the 9B baseline)

The 250M checkpoint fell to a combined score of `1.122947`, which is
`-0.009395` below baseline. The result supports checkpoint selection and early
stopping; it does not support treating longer continuation as monotonically
beneficial.

## Configuration

The experiment kept the complete Qwen3.5-9B host and inserted frozen donor
layers 31--33 from Qwen3.5-27B after host layer 15. The donor layers executed on
every forward pass. A trainable 512-wide, eight-head causal cross-attention
module fused donor information into the host stream, with a maximum residual
scale of `0.05`.

The host and original donor parameters remained frozen. Training was restricted
to the input/output projections and report-specific graft parameters. Each
checkpoint used the same protocol identity:
`b82fc1daf569d506e0a1ff24d1e96674879315248311badd10966164a685e111`.

Evaluation covered the complete deterministic task sets:

- MMLU-Pro: 12,032 samples
- BBH: 5,761 samples

The baseline and baseline-repeat evaluations were identical.

## Results

| Checkpoint | MMLU-Pro | BBH | Combined | Combined delta vs. baseline |
|---:|---:|---:|---:|---:|
| 9B baseline | 0.514046 | 0.618295 | 1.132341 | -- |
| 1M | 0.512799 | 0.617428 | 1.130227 | -0.002115 |
| 50M | 0.512716 | 0.616039 | 1.128755 | -0.003586 |
| **100M** | **0.514628** | **0.623503** | **1.138131** | **+0.005789** |
| 150M | 0.509475 | 0.616560 | 1.126034 | -0.006307 |
| 200M | 0.514295 | 0.618990 | 1.133285 | +0.000944 |
| 250M | 0.508644 | 0.614303 | 1.122947 | -0.009395 |

## Interpretation

The cross-attention graft can outperform the frozen host at a selected
checkpoint, driven mostly by BBH. The trajectory is unstable: the 100M gain
disappears at 150M, partially returns at 200M, and reverses at 250M. This makes
the 100M result evidence for a useful intermediate state, not evidence that the
architecture reliably improves with additional continuation.

Follow-up work should evaluate the sealed 100M checkpoint on the campaign's
held-out tasks and compare multiple seeds before making a broader capability
claim. Promotion rules should select on a fixed validation protocol and retain
the selected checkpoint rather than automatically using the final checkpoint.

## Artifact evidence

The local campaign artifacts are not committed to this repository. The hashes
below bind this report to their immutable completion receipts.

| Checkpoint | Training completion SHA-256 | Evaluation completion SHA-256 |
|---:|---|---|
| 1M | `880dbbef7173c3356a9b42ed50946f7856ea7dcfa8b752726a98d97b13be002b` | `d6c1b16825dac97fcd42e62cf9bc61ec84b13b05f850fe6488a29cc79e11dcf4` |
| 50M | `dc23d8d18c4179dc96e68729ff66538c8599fac47344875a15cd157d922800dd` | `bacd0deeb4aaf986b6cbb5338af0bac3d0709b0f00eab469da7af542a5ce21ba` |
| 100M | `ea07790362318cb0e59b7c6804d1d5ccf9bd33062d3516872fc299318f7a7c32` | `414b5821cddcefe0ae9766ce811b41040dfda4bbe27648c6443f834227edee60` |
| 150M | `f5c4f3f64cc971524064e9ac55a4e8ba10947f7a97aba0a569c85e0d19f6a6fb` | `f69f219eef484681e92949f59b9c4a24e52c765efb0f36492f8cc4c408c7d1b5` |
| 200M | `07da1faeeeacd8304d3292abc3c1bc85fdd4d22078958ada53ffd383b4b5d73b` | `9a4a96ef9f6fd7f8f9ffb6d8b9e79234b17208f3f67a0a5be5f279058a4cc2f7` |
| 250M | `1f408110f1680e234a426a0ade31c113cd311b3384096ecaeff3433de5eb933b` | `a4c0f9cda514bafe31dcf9128e80e27914875b9c9ddec2e112c0af765a48851c` |

The runtime receipt binds runner SHA-256
`e727e864c0703be6003a0e707983660e05d24cb1e2d64e56a7dcc2cbd9eaa950`
and core SHA-256
`a3cdcd11273b87ed6edbaf48990e304a915767a9de1c9cce7ec8d11fd6364b8a`.
