# CLAUSESHIFT — Technical Report

**Hack Apertus 2026 · Track 2A · UZH / SwissGov-RSD**
**Submission method:** `H1-HYBRID`, release V1.15.2 (frozen)
**Execution state:** completed; 672/672 prediction records validated (224 EN–DE, 224 EN–FR, 224 EN–IT). Final hidden performance is not known.
**Contributors:** see repository commit history and official submission form.

## 1. Problem and winning thesis

The SwissGov-RSD task is word-level cross-lingual semantic difference recognition. For each English text and its German, French or Italian counterpart, the model predicts a difference score for every whitespace-delimited token on both sides, retaining required masking conventions. Official development evaluation computes token-level Spearman rank correlations for the three language pairs and reports their macro mean. The goal is accountable token-local evidence, not a generative paraphrase of each document.

CLAUSESHIFT combines a public multilingual encoder with a small, bounded and independently observable Apertus contribution. A conservative fallback makes every invalid or missing remote response non-catastrophic. The method prioritizes exact token coverage and reproducible provenance over unsubstantiated score improvement claims.

## 2. Frozen architecture

1. **Input isolation.** The runner retrieves benchmark inputs at the pinned SwissGov-RSD revision and strips the downloaded rows immediately to `id`, `text_a`, `text_b`. Gold labels are neither consulted nor supplied to the model. Document order is preserved.
2. **Complete baseline.** `DiffAlign` from SwissGov-RSD evaluates public `jhu-clsp/mmBERT-base` (last layer, CPU) and produces per-token difference scores for both languages. For ordinary texts the full input is inferred in one pass.
3. **Long-input completion (V1.15.2).** If either source exceeds 3,072 subtokens, fixed parallel word windows with 24-word left/right context are processed, never exceeding 3,072 subtokens **per inference**. Central token scores are reconstructed exactly once, in original order; missing, non-finite or misaligned scores fail closed. This is an approximate bilingual window alignment, not a change to the learned model.
4. **Apertus selection.** Without reading any annotations, deterministic length-stratified selection picks 18 documents per language (54 total). Local window selection uses the baseline predictions and relative positions. At most 12 word indices on each side (24 total) are requested in a 32-word context per document.
5. **Provider invocation and fusion.** Model `swiss-ai/apertus-v1.5-8b` on `https://api.publicai.co/v1` returns explicit indexed similarities from 0 to 5. For accepted positions, `difference = 0.65 * mmBERT_difference + 0.35 * (1 - similarity / 5)`. Other positions retain mmBERT scores. Invalid responses or failed calls fail closed into baseline scores; no automatic retries occur. The maximum is one attempted request per selected document.
6. **Checkpoints and validation.** Baseline/plan and API results are recorded with deterministic signatures. A started remote request is not issued again after interruption. Each final export is checked for IDs, order, token counts, ranges and numerical finiteness before completion is reported.

**Pinned dependencies:** SwissGov-RSD Git commit `1807a42100e742ed03d337c54c4b9ea86995f565`; mmBERT Hugging Face revision `c5955035435e2bf121cde7f3c8863ef52ff35d82`. Frozen runner preflight digest prefix: `2d5d89cc996e`. Torch CPU and relevant Python dependencies are specified in `requirements.txt` and the Dockerfile; the Docker build was run successfully with Python 3.12. All scoring parameters and the runner are frozen by `docs/submission_frozen.json`.

## 3. Development measurements (not hidden evaluation)

DEV includes 168 documents per language (504 total). The public mmBERT model was evaluated on all 504 using the official DEV scoring script:

| Method and population | EN–DE | EN–FR | EN–IT | Macro Spearman |
| --- | ---: | ---: | ---: | ---: |
| Public mmBERT alone — entire DEV | 0.146786 | 0.087821 | 0.152947 | **0.129185** |
| Public mmBERT alone — H1-selected sample (18 per language) | 0.126881 | 0.065680 | 0.127844 | 0.106802 |
| H1 hybrid — same selected sample | 0.128646 | 0.063163 | 0.131595 | **0.107801** |
| H1 − baseline on selected sample | +0.001766 | −0.002517 | +0.003751 | **+0.001000** |

The 54-document ablation attempted 54 Apertus calls: **44 accepted, 10 fallback**; 1,056 selected token scores changed. The macro improvement is very small and FR regressed. The selected-document macro is not interchangeable with the full-DEV macro. No official full-DEV hybrid result was established and no confidence interval demonstrating robust improvement is claimed. Prior literature reporting a fine-tuned mmBERT score of approximately **0.2213** uses different adapted weights; this project uses the public base checkpoint.

Earlier E0-PJQ and E2 local exploratory single-document experiments are preserved in the source tree for research provenance, but did not determine the final hybrid parameters from final/test gold labels. Their one-document proxy scores must not be represented as official benchmarks.

## 4. Final execution and validity

The frozen V1.15.2 Docker runner completed **224 records each for DE, FR and IT**. The Docker offline validator confirmed all **672 records** with matching IDs, original order, corresponding token counts and in-range scores. Copies of the final outputs passed independent SHA-256 integrity checks. The outputs are:

- `out/predictions/CLAUSESHIFT_admin_de.jsonl.jsonl` — SHA-256 `97b81c8b33a0be09d1d7a33ca8add6ba975130c3ef5d0f8dcc2ec7278827164a`.
- `out/predictions/CLAUSESHIFT_admin_fr.jsonl.jsonl` — SHA-256 `a97df1757ceb15137c003736005ce70ea9430092cd3a73b2d7cac1227a47dee7`.
- `out/predictions/CLAUSESHIFT_admin_it.jsonl.jsonl` — SHA-256 `320692e348029c23bcdf308d722b00b177a4f370ba153910022082f2dfd6ecb1`.

All reported final checks are **structural checks**, not Spearman evaluation. No final gold annotations were used to measure quality or select a method.

## 5. Freeze history and integrity incident

V1.15.1 stopped deliberately when a long final input exceeded the original 3,072-subtoken full-document cap. There was no fabricated padding or truncated exported prediction. After archiving the earlier freeze, V1.15.2 was implemented and re-frozen to add generic, bounded long-input segmentation. Documents inside the cap preserve the original inference path; the change is solely to achieve complete coverage and may nevertheless change the scores of long documents. It was **not** designed by optimizing final-set Spearman or inspecting gold labels. The earlier and later freeze points are represented in the local Git history and the committed `docs/submission_frozen.json`; see `docs/FROZEN_H1_V1152_FR.md` for the chronology.

**Competition-policy caveat:** because this completeness fix followed a first access to final inputs, its permissibility depends on the organizers' rules; this is disclosed, not assumed. The final method does not claim a performance improvement attributable to the hotfix.

## 6. Runtime, reproducibility and limitations

The end-to-end run was tested on an Ubuntu 24.04 DigitalOcean VPS with **8 Intel vCPUs, 16 GiB RAM, CPU-only execution**, using Docker with Python 3.12. The exact Docker image and model load were verified on the VPS. The runner requires internet access to fetch pinned public resources and to reach an OpenAI-compatible Apertus provider. Dependencies are pinned in `requirements.txt`; the external service remains a reproducibility limitation because its availability, quotas, exact served model revision and outputs are not fully controlled by the project.

Required shell environment variables:

```bash
export LLM_NAME='swiss-ai/apertus-v1.5-8b'
export LLM_BASE_URL='https://api.publicai.co/v1'
export LLM_API_KEY='<provide as an environment secret; never commit>'
```

From repository root: `make run` should call the `track_2a/` runner with model warmup. Directly from `track_2a/`: `make warmup && make run`. The nested `make run` rebuilds Docker, retrieves text-only inputs and produces three output files. Re-running requires a reachable provider and may not reproduce identical remote judgments; checkpoints avoid repeating previous attempts only when reused. The exact final prediction files and hashes listed above are the retained execution artifacts and must not be confused with independent fresh-run outputs.

The public encoder establishes a usable baseline but did **not** reproduce fine-tuned research performance. A targeted 54-document ablation is insufficient to establish statistically robust model improvement. Proportional segmentation can reduce cross-language contextual alignment near boundaries. The method was developed without a held-out final quality estimate. Credential management and limits on API calls are external deployment responsibilities.

## 7. Licenses and references

Code is released under **Apache-2.0**. Original project documentation and this technical report are provided under **CC BY 4.0** insofar as the authors hold the applicable rights. Third-party benchmark data, model weights and research materials retain their respective licenses and attributions; they are not republished with the code.

- [Hack Apertus project template](https://github.com/HackApertus/project-template)
- [Hack Apertus 2026 submission portal](https://hackapertus.ch/online-hack/submissions)
- [SwissGov-RSD research repository](https://github.com/ZurichNLP/SwissGov-RSD)
- [mmBERT-base model](https://huggingface.co/jhu-clsp/mmBERT-base)
- [SwissGov-RSD ACL paper](https://aclanthology.org/2026.acl-long.1437.pdf)
