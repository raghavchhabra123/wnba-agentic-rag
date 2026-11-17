# Descriptive-answer faithfulness

45 descriptive questions over 15 games. Claims checked against the full play-by-play and box score by an LLM judge (validated, see judge_validation.json).

| System | Claims | Supported | Contradicted | Not verifiable | Answers with ≥1 false claim |
|---|---:|---:|---:|---:|---:|
| naive_rag | 236 | 68.2% | 27.5% | 4.2% | 71% |
| scoped_rag | 386 | 77.2% | 18.9% | 3.9% | 78% |
| agentic | 465 | 83.7% | 14.8% | 1.5% | 64% |
