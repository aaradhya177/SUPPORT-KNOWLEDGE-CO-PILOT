# Retrieval Accuracy Comparison

Evaluation examples: 18

Metric: Hit Rate@5, counted as a hit when the expected chunk, expected document, or expected support evidence keyword appears in the top results.

| Retriever | Hit Rate@5 | Hits | Total |
| --- | --- | --- | --- |
| Dense | 1.0000 | 18 | 18 |
| BM25 | 1.0000 | 18 | 18 |
| Hybrid RRF | 1.0000 | 18 | 18 |

## Interpretation

The strongest retriever in this run is `Dense` with 100.00% Hit Rate@5. Hybrid retrieval is expected to outperform either individual retriever when the eval set mixes exact lexical needs and paraphrased intent: BM25 is strong for precise terms such as HTTP 429, invoice, and tax exemption, while dense retrieval can recover semantically similar support questions that do not reuse the document wording. The reported numbers are computed from this eval set and the currently built indexes; they should be rerun whenever the corpus, chunking strategy, embedding model, or fusion settings change.

## Resume Claim Guidance

Do not claim a retrieval lift such as `72% to 88%` unless this report actually shows those measured values on a sufficiently large and non-trivial eval set. When the corpus is tiny or the number of indexed chunks is less than or close to the Hit Rate cutoff, Hit Rate@K can saturate and stop distinguishing retrievers.
