# Evaluation results (dev split)

223 questions from 25 games; deterministic grading (see eval/grade.py). Model: gpt-4o-mini for every system.

| System | Accuracy | Correct | Median latency (s) |
|---|---:|---:|---:|
| llm_only | 0.0% | 0/223 | 1.4 |
| naive_rag | 23.8% | 53/223 | 1.8 |
| scoped_rag | 37.2% | 83/223 | 3.9 |
| agentic | 100.0% | 223/223 | 3.0 |

## Accuracy by question type

| Question type | llm_only | naive_rag | scoped_rag | agentic |
|---|---:|---:|---:|---:|
| final_score | 0% | 40% | 32% | 100% |
| first_three | 0% | 8% | 24% | 100% |
| largest_lead | 0% | 8% | 16% | 100% |
| last_points | 0% | 32% | 56% | 100% |
| player_points_box | 0% | 4% | 0% | 100% |
| q4_top_scorer | 0% | 40% | 36% | 100% |
| quarter_winner | 0% | 32% | 52% | 100% |
| second_half_turnovers | 0% | 39% | 70% | 100% |
| team_leader_box | 0% | 12% | 52% | 100% |

## Game resolution

- agentic: resolved the correct game for 223/223 questions
- scoped_rag: resolved the correct game for 223/223 questions
