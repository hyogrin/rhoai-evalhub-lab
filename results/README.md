# Evaluation Results

This directory stores evaluation results from EvalHub runs on OpenShift AI.

## Result Format

Results are stored per model in JSON format:

```
results/
├── glm-53-flash/
│   ├── unified-kmmlu-0921-1500_abc12345.json
│   ├── unified-click-0921-1500_abc12345.json
│   └── ...
├── qwen3-14b/
│   ├── kmmlu-eval_54c2d994.json
│   └── ...
└── RESULTS.md
```

Each JSON file contains the EvalHub job output including:
- `job_id`: EvalHub job identifier
- `model`: Model endpoint and name
- `experiment`: MLflow experiment name
- `benchmarks`: Benchmark results with metrics

## Extracting Results from EvalHub

```python
# Via EvalHub SDK
from evalhub import SyncEvalHubClient
client = SyncEvalHubClient(base_url=EVALHUB_URL, auth_token=TOKEN, insecure=True, tenant=NAMESPACE)
job = client.jobs.get(job_id)
print(job.results.benchmarks[0].metrics)
```

## Comprehensive Results

For accumulated benchmark results across major open-weight models (GPT, Gemma, Llama, Phi, Qwen, etc.) with radar chart visualizations, see:

> **[evaluate-llm-on-korean-dataset](https://github.com/hyogrin/evaluate-llm-on-korean-dataset)**

That repository tracks:
- KMMLU (45 subjects)
- CLIcK (11 categories)
- HAE-RAE (6 categories)
- HRM8K (math reasoning)
- KoBALT (linguistic phenomena)
- KorMedMCQA (medical QA)
