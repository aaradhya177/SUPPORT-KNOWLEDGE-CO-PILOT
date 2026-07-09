# Golden Set Evaluation Summary

Questions evaluated: 10

| Metric | Value |
| --- | --- |
| Retrieval Hit Rate | 0.9000 |
| Avg Answer Correctness | 0.9400 |
| Avg Citation Faithfulness | 0.9000 |
| No-Answer Precision | 1.0000 |
| No-Answer Recall | 1.0000 |

## Interpretation

Retrieval hit rate measures whether at least one expected source document appeared in the retrieved context. Answer correctness is LLM-graded against the human-authored expected summary. Citation faithfulness measures how many judge verdicts were fully supported. No-answer precision and recall specifically evaluate whether the system refuses plausible but unsupported questions without over-refusing answerable ones.
