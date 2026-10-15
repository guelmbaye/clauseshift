## V1.15 H1 hybrid — measured evidence and limits (2026-10-15)

The production candidate is a fixed fusion of publicly available `jhu-clsp/mmBERT-base` DiffAlign scores with indexed Apertus v1.5 8B local-window adjudication. For accepted token scores, use `0.65×encoder + 0.35×(1−Apertus_similarity/5)`; all others retain encoder predictions. Exactly 18 length-stratified documents per language and at most 54 calls, one per selected document. All choices are determined from input texts and local baseline scores (no annotations). An invalid/aborted API attempt never triggers automatic retries.

**DEV baseline**, full 168 documents/language, evaluated with SwissGov-RSD official evaluator: Spearman DE 0.146786 / FR 0.087821 / IT 0.152947, macro 0.129185. **H1 ablation**, selected 18/language only, 54 attempts, 44 accepted, 10 fallback, 1056 token scores changed: baseline 0.106802, hybrid 0.107801 (+0.001000); DE +0.001766, FR −0.002517, IT +0.003751. This is a marginal exploratory benefit on a selected subset, **not** evidence of a robust global improvement. Other researchers' published mmBERT fine-tuned baseline (0.2213) is NOT the model in this delivery.

The final submission runner accepts only unlabeled input rows and is designed to refuse incomplete/malformed output. The model revision and upstream recognizer commit are pinned. The V1.15 image build and real end-to-end runtime still require independent verification on the target VPS; scores on final data are unknown.

---

# Technical report — CLAUSESHIFT

- **Track:** 2A — UZH, SwissGov-RSD
- **Event:** Hack Apertus 2026 Online Hackathon
- **Team:** To fill in before submission
- **Demo:** Optional; not yet recorded
- **State:** E0-PJQ/E2 experiments completed on a few DEV documents; no official full DEV score or finalized submission yet

## 1. Summary
CLAUSESHIFT aims to detect token-level cross-lingual differences on SwissGov-RSD using Apertus 1.5. The implementation includes a strict, resumable E0-PJQ inference pipeline and separate E2 DEV-only alignment experiments. Real Apertus calls have run on selected DEV documents only; no benchmark-wide gain is claimed.

## 2. Architecture
See `docs/DOCUMENT_01_FROZEN_V1.2.md` and `src/`. Input-only dev JSONL -> paper-style or exact paper Appendix-E prompt -> Apertus inference -> strict token projections -> encoder-compatible JSONL -> official evaluator.

## 3. Use of Apertus
- **Model:** `swiss-ai/apertus-v1.5-8b` through the Public AI OpenAI-compatible endpoint (DEV runs); version/revision must be explicitly documented in final manifest.
- **Method:** Inference with deterministic/temperature-zero decoding if supported.
- **Endpoint:** Configured by `LLM_BASE_URL`, credentials in `LLM_API_KEY`.
- **Prompt:** `PAPER_STYLE` in `src/core.py` is not the exact research prompt; for verified reproduction supply `prompts/paper_exact.txt`, containing full Appendix E.

## 4. Data
SwissGov-RSD, public benchmark. DEV only until code freeze. `data/inputs` contains unlabeled text and IDs only. Respect copyright and CC BY licence information in the research paper/dataset.

## 5. Evaluation
Primary token-level Spearman, scored separately for EN–DE, EN–FR and EN–IT, macro-averaged. Official script `scripts.evaluate_predictions_admin` with `--split dev` during development. Exploratory matched-document results only, not official benchmark-wide Spearman.

| Setup | Mean Spearman (dev) | EN-DE | EN-FR | EN-IT | Runtime |
|---|---|---|---|---|---|
| E0-PJQ: paper few-shot + conservative repairs | 0.1561* | 0.4771 | -0.0293 | 0.0203 | Not measured for full set |
| E2 Coverage V1.10: semantic align + few-shot indexed | 0.1138* | 0.2911 | 0.1262 | -0.0759 | Not measured for full set |

*Proxy Spearman on exactly one overlapping DEV document (`admin_*_105`) per language; **NOT** an official macro-score, robust confidence estimate or held-out result. E2 outperforms E0-PJQ on FR only. On `admin_de_27`, a different E2 variant scored 0.3225 vs E0-PJQ 0.2442, further showing the instability of small-sample comparisons.*

## 6. Limitations
Benchmark-wide evaluation and complete execution are pending. Apertus was called on selected DEV documents. Long documents may exceed CSCS's reported 60-second gateway limit; invalid/incomplete results are rejected, not silently padded.

## 7. Reproducibility
With V1.11, `make run` is provisionally wired to frozen E0-PJQ and a strict Docker validation path; until a real, committed freeze manifest exists it intentionally fails closed. The runner has not been tested on the full benchmark or real Docker and is not proof of final completion. Dev/final split gates implemented in `src/cli.py`. Docker runtime unavailable in package build environment; must be tested externally.

## 8. Next steps
1. Run real E0 on dev with exact paper prompt.
2. Reproduce evaluation and report scores/costs.
3. Compare E1 vs E2, record ablations.
4. Freeze, final full-order predictions, test only after freeze.

## License
Technical report CC-BY-4.0. Code Apache-2.0.

## References
- https://aclanthology.org/2026.acl-long.1437.pdf
- https://github.com/ZurichNLP/SwissGov-RSD
- https://github.com/HackApertus/project-template/tree/main/track_2a

### E1-C experimental candidate — development only
A separate chunked method with proportional token windows, bilingual context and numeric-only JSON responses is implemented. It must not be conflated with the published prompting baseline. No SwissGov-RSD performance result has yet been observed. A DEV-only exploratory matched-ID Spearman helper is included, and its result is not the official competition metric. Final submission integration and Docker reproduction remain to be implemented after a methodology is selected and frozen.

### Experimental candidate E2 (evaluated on a few DEV records)

Separate DEV-only experiments E2-A (semantic-aligned/short indexed), E2-B
(proportional/few-shot indexed) and E2-C (semantic-aligned/few-shot indexed) were
implemented as candidate workflows. Semantic alignment relies on an auxiliary
multilingual embedding model, dynamic-programming monotonic matches with
explicit unmatched spans, and strict token-preserving reconstruction. Unmatched
content receives model-generated scores, never scores fabricated by the planner.
The indexed few-shot instruction is adapted from Appendix E and is **not**
claimed to be an exact paper baseline. E2 V1.10 achieved 0.2911/0.1262/-0.0759 Spearman proxy on `admin_*_105` for EN–DE/FR/IT, respectively. No full benchmark or held-out test performance has been measured. The final model selection and
frozen held-out test execution are pending.


## V1.14 multi-unit batching — experimental method and limits

The new candidate groups consecutive, coverage-aware aligned bilingual units
into calls capped by semantic score count and a unit-count limit. Prompt output
is indexed per unit; parser rejects missing, duplicate, unexpected and invalid
indices. Each accepted batch is saved in signed append-only checkpoints.
Original token ordering is rebuilt before any predictions are exported.
Concurrent workers are bounded and checkpoint writes serialized. Final inputs
are prohibited until signed submission preflight.

The batching modifies the model's context and is not equivalent to E0-PJQ or
the previously evaluated E2 variant. Its Spearman quality is **unmeasured**;
all tests are synthetic and the end-to-end Docker run has not been tested.
Existing E0 and E2 comparisons are small, exploratory, and not UZH scores.

### Submission hotfix V1.15.2 — coverage only, after first frozen run

The original V1.15.1 runner failed closed on a final input exceeding the conservative 3072-subtoken **per-call** limit. After that failure, V1.15.2 added deterministic proportional word-based parallel windows with 24-token overlaps per side. Each original token's score is taken from the single corresponding core window, with strict token-order and output-length validation. No padding/truncation or synthetic similarity scores are used. Documents inside the original cap follow the exact original single-call path. This is a **post-freeze technical hotfix**, not a measured improvement to DEV Spearman, and any competition freeze restrictions must be checked and disclosed. All reported DEV/H1 metrics refer to the earlier measured methods and are not new V1.15.2 scores.
