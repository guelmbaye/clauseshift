# CLAUSESHIFT — Method visualization

Hack Apertus 2026 · Track 2A · SwissGov-RSD · H1-HYBRID V1.15.2

```mermaid
flowchart LR
    A[English + German / French / Italian text] --> B[Public mmBERT-base / DiffAlign]
    B --> C{Input exceeds 3,072 subtokens?}
    C -- No --> D[One-pass word-level scoring]
    C -- Yes --> E[Overlapping bilingual windows / 24 context words]
    E --> F[Reassemble each original word score]
    D --> G[Full-document baseline]
    F --> G
    G --> H[Deterministic 18 docs per language selection]
    H --> I[Up to 54 Apertus 1.5 8B calls]
    I --> J{Valid indexed response?}
    J -- Yes --> K[Blend selected token scores / 0.65 mmBERT + 0.35 Apertus difference]
    J -- No --> L[Retain baseline scores]
    K --> M[Validate order, token counts, ranges]
    L --> M
    G --> M
    M --> N[DE / FR / IT JSONL predictions]
```

**Bounded external contribution:** 18 selected documents per language, up to 12 tokens on each side within 32-word windows, at most 54 calls. Invalid responses fail closed without retry.

**Measured development evidence:** public mmBERT all-DEV macro Spearman = **0.129185**. On a 54-document *selected* DEV ablation, H1 − mmBERT = **+0.001000** macro, with French regression. These are not final test results.

**Final structural validation:** 224 EN–DE + 224 EN–FR + 224 EN–IT = **672/672 records** validated by the offline Docker validator. This measures formatting and coverage, not hidden-set accuracy.

For full reproducibility, limitations and disclosure of the post-first-access completeness fix, consult [the technical report](../technical_report.md).
