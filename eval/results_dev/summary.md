# Evaluation results: v1 dev split

223 questions; deterministic grading (see eval/grade.py). Model: gpt-4o-mini for every system.

| System | Accuracy | Correct | Median latency (s) | Avg cost / question |
|---|---:|---:|---:|---:|
| llm_only | 0.0% | 0/223 | 1.4 | - |
| naive_rag | 23.8% | 53/223 | 1.8 | - |
| scoped_rag | 37.2% | 83/223 | 3.9 | - |
| agentic | 99.1% | 221/223 | 3.6 | $0.00066 |

## Accuracy by question type

| type | n | llm_only | naive_rag | scoped_rag | agentic |
|---|---:|---:|---:|---:|---:|
| final_score | 25 | 0% | 40% | 32% | 100% |
| first_three | 25 | 0% | 8% | 24% | 96% |
| largest_lead | 25 | 0% | 8% | 16% | 100% |
| last_points | 25 | 0% | 32% | 56% | 96% |
| player_points_box | 25 | 0% | 4% | 0% | 100% |
| q4_top_scorer | 25 | 0% | 40% | 36% | 100% |
| quarter_winner | 25 | 0% | 32% | 52% | 100% |
| second_half_turnovers | 23 | 0% | 39% | 70% | 100% |
| team_leader_box | 25 | 0% | 12% | 52% | 100% |

## Game resolution (single-game questions)

- agentic: resolved the correct game for 223/223 questions
- scoped_rag: resolved the correct game for 223/223 questions
