# CLAUSESHIFT — Hack Apertus 2026, Track 2A (UZH)

**Evidence-grounded cross-lingual semantic difference recognition with Apertus 1.5 and mmBERT/DiffAlign.**

**Submission status:** V1.15.2 inference completed on 224 EN–DE, 224 EN–FR, and 224 EN–IT records (672 records total). All three prediction files passed strict offline validation of document order, IDs, token counts and score ranges. SHA-256 integrity checks also passed. **No held-out final Spearman score is known or claimed.**

## Approach

The frozen `H1-HYBRID` runner computes word-level difference estimates using the public `jhu-clsp/mmBERT-base` encoder via SwissGov-RSD `DiffAlign` (`layer=-1`). It then requests targeted word-level similarity judgments from `swiss-ai/apertus-v1.5-8b` for up to 18 deterministically selected documents per language (54 requests maximum). Accepted scores are fused as `0.65 * mmBERT_difference + 0.35 * (1 - Apertus_similarity / 5)` at the requested token positions. Invalid or unavailable responses fall back to the mmBERT scores; automatic retries are disabled. Unselected tokens retain mmBERT scores.

V1.15.2 processes over-length documents by overlapping, bilingual word windows with 24 context words per side and a strict limit of 3,072 subtokens **per mmBERT inference**. Every output token is sourced from exactly one core passage. Short documents retain the unsegmented path. The fallback and segmented-passage methods do **not** imply guaranteed quality improvement.

## Measured development results

| Measurement | EN–DE | EN–FR | EN–IT | Macro |
| --- | ---: | ---: | ---: | ---: |
| Public mmBERT baseline on all 168 DEV documents/language | 0.146786 | 0.087821 | 0.152947 | **0.129185** |
| mmBERT on selected 18 DEV documents/language | 0.126881 | 0.065680 | 0.127844 | 0.106802 |
| H1 hybrid on the *same selected* DEV documents | 0.128646 | 0.063163 | 0.131595 | **0.107801** |

The hybrid difference on the selected DEV sample was **+0.001000** macro. Of 54 requests, 44 were accepted and 10 fell back; 1,056 scores were changed. The French result **decreased**. This small selected-sample improvement is **not** an official full-DEV hybrid score or a reliable estimate of held-out performance. The previously reported fine-tuned mmBERT score 0.2213 uses **different weights**, not the public encoder used here.

## Reproduction

Prerequisites: Docker Engine, internet access to GitHub/Hugging Face/Public AI, sufficient RAM for mmBERT inference (tested on a 16 GiB CPU VPS), and valid Apertus provider credentials. Running the method entails paid or rate-limited API calls; provider responses may differ between runs.

Set the following environment variables in your shell (do **not** commit credentials):

```bash
export LLM_NAME='swiss-ai/apertus-v1.5-8b'
export LLM_BASE_URL='https://api.publicai.co/v1'
read -r -s -p 'Apertus API key: ' LLM_API_KEY; echo
export LLM_API_KEY
```

From the **repository root**, the provided root `Makefile` should execute `make warmup` and `make run` within `track_2a/`:

```bash
make run
```

Alternatively, from this directory:

```bash
make warmup  # Download pinned public mmBERT weights to .hf_cache/
make run     # Build Docker; retrieve unlabeled text fields and generate predictions
```

Prediction filenames (relative to `track_2a/`):

```text
out/predictions/CLAUSESHIFT_admin_de.jsonl.jsonl
out/predictions/CLAUSESHIFT_admin_fr.jsonl.jsonl
out/predictions/CLAUSESHIFT_admin_it.jsonl.jsonl
```

`make preflight` checks the frozen manifest without accessing final inputs; `python -m unittest discover -s tests -q` runs the included unit tests. No final gold labels or final-set metric should be read, computed or used for model selection. Source data are retrieved from pinned SwissGov-RSD files and immediately stripped to `id`, `text_a`, `text_b` before inference.

Pinned public artifacts:

- SwissGov-RSD recognizer commit: `1807a42100e742ed03d337c54c4b9ea86995f565`.
- mmBERT weights revision: `c5955035435e2bf121cde7f3c8863ef52ff35d82`.
- V1.15.2 frozen code digest: `2d5d89cc996e` (prefix from preflight).

## Transparency and constraints

Following the V1.15.1 frozen run, a final input exceeded the conservative 3,072-subtoken full-document limit; the runner stopped without producing incomplete outputs. V1.15.2 introduced the generic long-document segmentation described above and was **re-frozen**. This post-freeze completeness change was motivated by input length, not by consulting final annotations or scores. Its admissibility remains subject to the competition's rules; see `docs/FROZEN_H1_V1152_FR.md` and the technical report. No new DEV performance claim is attached to this change.

The project also retains earlier E0/E1/E2 research prototypes in `src/`, `tools/`, and `docs/`; these are **not** the frozen submission path. The final method is `H1-HYBRID`.

## Attribution and license

Code: Apache-2.0. Technical report and original project documentation: CC BY 4.0, subject to third-party rights. SwissGov-RSD, mmBERT, Apertus and underlying source texts retain their respective upstream licenses. See `technical_report.md` for methods, results, limitations, infrastructure, and references.

## Submitted prediction artifacts

The three validated final prediction files are committed
to this repository under `predictions/`:

- `predictions/CLAUSESHIFT_admin_de.jsonl.jsonl`
- `predictions/CLAUSESHIFT_admin_fr.jsonl.jsonl`
- `predictions/CLAUSESHIFT_admin_it.jsonl.jsonl`

These are the frozen V1.15.2 submission artifacts
(224 records per language, 672 records total).

A fresh `make run` generates predictions under
`out/predictions/`. This runtime directory is Git-ignored,
while `predictions/` contains the committed submission copies.

The validated SHA-256 hashes are recorded in
`technical_report.md`.
