"""
Korean LLM Evaluation Pipeline — KFP v2 on RHOAI 3.5 Data Science Pipelines.

5-step pipeline using the evalhub SDK (same as 3_eval_hub_unified_benchmark notebooks):
  1. setup        — Verify EvalHub connectivity and Korean MCQ provider
  2. accuracy     — Submit 5 Korean benchmarks (parallel with step 3)
  3. performance  — GuideLLM throughput benchmark (parallel with step 2)
  4. reports      — Generate report.html + RESULTS.md
  5. notify       — Send Slack webhook notification

Usage:
  python compile.py
  Then import eval_pipeline.yaml via RHOAI Dashboard → Pipelines → Import pipeline.
"""

from kfp import dsl
from kfp.dsl import Input, Output, Artifact, Dataset

BASE_IMAGE = "registry.redhat.io/ubi9/python-311:latest"
SDK_PACKAGES = ["eval-hub-sdk[client]==1.0.5"]


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Setup & Verify
# ──────────────────────────────────────────────────────────────────────────────
@dsl.component(base_image=BASE_IMAGE, packages_to_install=SDK_PACKAGES)
def setup_and_verify(
    evalhub_url: str,
    auth_token: str,
    namespace: str,
    provider_name: str,
) -> str:
    """Verify EvalHub connectivity and return the Korean MCQ provider ID."""
    from evalhub import SyncEvalHubClient

    client = SyncEvalHubClient(
        base_url=evalhub_url,
        auth_token=auth_token,
        insecure=True,
        tenant=namespace,
    )

    health = client.health()
    print("Health: " + str(health))

    providers = client.providers.list()
    print("Providers: " + str(len(providers)))

    for p in providers:
        print("  - " + p.name + " (" + p.resource.id + ")")
        if p.name == provider_name:
            print("Match: " + provider_name + " -> " + p.resource.id)
            return p.resource.id

    # Partial match fallback
    for p in providers:
        if provider_name.lower() in p.name.lower():
            print("Match (partial): " + p.name + " -> " + p.resource.id)
            return p.resource.id

    raise RuntimeError("Provider not found: " + provider_name)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Korean MCQ Accuracy (5 benchmarks)
# ──────────────────────────────────────────────────────────────────────────────
@dsl.component(base_image=BASE_IMAGE, packages_to_install=SDK_PACKAGES)
def run_accuracy_benchmarks(
    evalhub_url: str,
    auth_token: str,
    namespace: str,
    model_endpoint: str,
    model_name: str,
    model_api_key: str,
    provider_id: str,
    limit: int,
    experiment_name: str,
    results: Output[Dataset],
):
    """Submit 5 Korean MCQ benchmarks and poll until all complete."""
    import json
    import time

    from evalhub import (
        SyncEvalHubClient,
        ModelConfig,
        BenchmarkConfig,
        JobSubmissionRequest,
        ExperimentConfig,
    )

    client = SyncEvalHubClient(
        base_url=evalhub_url,
        auth_token=auth_token,
        insecure=True,
        tenant=namespace,
    )

    model = ModelConfig(url=model_endpoint, name=model_name)
    if model_api_key:
        from evalhub.models.api import ModelAuth
        model.auth = ModelAuth(secret_ref=model_api_key)

    benchmarks = ["click", "haerae", "kmmlu", "kmmlu_hard", "kobest_boolq"]
    job_ids = []

    for bm in benchmarks:
        params = {"temperature": 0.0, "max_tokens": 1024}
        if limit > 0:
            params["limit"] = limit

        req = JobSubmissionRequest(
            name="pipeline-" + bm + "-" + model_name,
            description="Pipeline accuracy: " + bm,
            tags=["pipeline", "accuracy", model_name],
            model=model,
            benchmarks=[
                BenchmarkConfig(
                    id=bm,
                    provider_id=provider_id,
                    parameters=params,
                )
            ],
            experiment=ExperimentConfig(name=experiment_name),
        )
        job = client.jobs.submit(req)
        job_ids.append((bm, job.id))
        print("  [" + bm + "] submitted: " + job.id)

    print(str(len(job_ids)) + " accuracy jobs submitted. Polling...")

    # Poll until all complete (max 45 min)
    all_results = {}
    for i in range(180):
        states = {}
        for bm, jid in job_ids:
            j = client.jobs.get(jid)
            states[bm] = j.state.value

        status_str = " | ".join(k + "=" + v for k, v in states.items())
        if i % 4 == 0:
            print("  [" + str(i * 15) + "s] " + status_str)

        if all(s in ("completed", "failed", "cancelled", "partially_failed") for s in states.values()):
            print("All accuracy jobs finished!")
            break
        time.sleep(15)

    # Collect results
    for bm, jid in job_ids:
        j = client.jobs.get(jid)
        bm_data = {"state": j.state.value, "metrics": {}, "mlflow_run_id": None}
        if hasattr(j, "results") and j.results:
            for r in j.results.benchmarks:
                bm_data["metrics"] = dict(r.metrics) if r.metrics else {}
                bm_data["mlflow_run_id"] = r.mlflow_run_id
                acc = r.metrics.get("overall_accuracy", "N/A")
                print("  " + bm + " = " + str(acc) + "%")
        all_results[bm] = bm_data

    output = {
        "model": model_name,
        "experiment": experiment_name,
        "benchmarks": all_results,
    }
    with open(results.path, "w") as f:
        json.dump(output, f, indent=2)
    print("Accuracy results saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: GuideLLM Performance
# ──────────────────────────────────────────────────────────────────────────────
@dsl.component(base_image=BASE_IMAGE, packages_to_install=SDK_PACKAGES)
def run_performance_benchmark(
    evalhub_url: str,
    auth_token: str,
    namespace: str,
    model_endpoint: str,
    model_name: str,
    experiment_name: str,
    results: Output[Dataset],
):
    """Run GuideLLM throughput benchmark via EvalHub SDK."""
    import json
    import time

    from evalhub import (
        SyncEvalHubClient,
        ModelConfig,
        BenchmarkConfig,
        JobSubmissionRequest,
        ExperimentConfig,
    )

    client = SyncEvalHubClient(
        base_url=evalhub_url,
        auth_token=auth_token,
        insecure=True,
        tenant=namespace,
    )

    # GuideLLM needs /v1 base endpoint
    guidellm_url = model_endpoint.rstrip("/")
    for suffix in ["/v1/completions", "/v1/chat/completions"]:
        guidellm_url = guidellm_url.replace(suffix, "/v1")
    if not guidellm_url.endswith("/v1"):
        guidellm_url += "/v1"

    perf_request = JobSubmissionRequest(
        name="pipeline-perf-" + model_name,
        description="GuideLLM throughput for " + model_name,
        tags=["pipeline", "performance", "guidellm", model_name],
        model=ModelConfig(url=guidellm_url, name=model_name),
        benchmarks=[
            BenchmarkConfig(
                id="throughput",
                provider_id="guidellm",
                parameters={
                    "profile": "throughput",
                    "max_seconds": 180,
                    "max_requests": 50,
                    "data": "prompt_tokens=128,output_tokens=64",
                    "request_type": "chat_completions",
                },
            )
        ],
        experiment=ExperimentConfig(name=experiment_name),
    )

    perf_job = client.jobs.submit(perf_request)
    print("GuideLLM job submitted: " + perf_job.id)

    # Poll (max 15 min)
    perf_metrics = {}
    state = "unknown"
    for i in range(90):
        j = client.jobs.get(perf_job.id)
        state = j.state.value
        if state in ("completed", "failed", "cancelled", "partially_failed"):
            print("GuideLLM " + state)
            if hasattr(j, "results") and j.results:
                for bm in j.results.benchmarks:
                    perf_metrics = dict(bm.metrics) if bm.metrics else {}
                    for k, v in sorted(perf_metrics.items()):
                        print("  " + k + ": " + str(v))
            break
        if i % 3 == 0:
            print("  [" + str(i * 10) + "s] " + state)
        time.sleep(10)

    output = {
        "model": model_name,
        "state": state,
        "metrics": perf_metrics,
    }
    with open(results.path, "w") as f:
        json.dump(output, f, indent=2)
    print("Performance results saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Generate Reports
# ──────────────────────────────────────────────────────────────────────────────
@dsl.component(base_image=BASE_IMAGE)
def generate_reports(
    accuracy_results: Input[Dataset],
    performance_results: Input[Dataset],
    report_html: Output[Artifact],
    results_md: Output[Artifact],
):
    """Generate report.html (Chart.js) and RESULTS.md from evaluation results."""
    import json
    from datetime import datetime

    acc_data = json.load(open(accuracy_results.path))
    perf_data = json.load(open(performance_results.path))

    model_name = acc_data.get("model", "unknown")
    benchmarks = acc_data.get("benchmarks", {})

    score_table = {}
    for bm_id, bm_result in benchmarks.items():
        metrics = bm_result.get("metrics", {})
        overall = metrics.get("overall_accuracy")
        if overall is not None:
            score_table[bm_id] = round(float(overall), 2)

    perf_metrics = perf_data.get("metrics", {})

    # --- RESULTS.md ---
    md_lines = ["# Evaluation Results", ""]
    md_lines.append("Model: " + model_name)
    md_lines.append("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    md_lines.append("")
    md_lines.append("## Accuracy")
    md_lines.append("")
    md_lines.append("| Benchmark | Accuracy |")
    md_lines.append("|:---|---:|")
    for bm_id in sorted(score_table.keys()):
        md_lines.append("| " + bm_id + " | " + str(score_table[bm_id]) + "% |")
    md_lines.append("")

    if perf_metrics:
        md_lines.append("## Performance")
        md_lines.append("")
        md_lines.append("| Metric | Value |")
        md_lines.append("|:---|---:|")
        for k in ["output_tokens_per_second", "prompt_tokens_per_second", "requests_per_second"]:
            v = perf_metrics.get(k)
            if v is not None:
                md_lines.append("| " + k + " | " + str(round(float(v), 2)) + " |")
        md_lines.append("")

    with open(results_md.path, "w") as f:
        f.write("\n".join(md_lines))
    print("RESULTS.md generated")

    # --- report.html ---
    tasks_sorted = sorted(score_table.keys())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    overall_avg = round(sum(score_table.values()) / len(score_table), 2) if score_table else 0

    html = '<!DOCTYPE html><html lang="ko"><head>'
    html += '<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">'
    html += '<title>Korean LLM Benchmark Report</title>'
    html += '<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>'
    html += "<style>"
    html += "* { margin: 0; padding: 0; box-sizing: border-box; }"
    html += "body { font-family: -apple-system, sans-serif; background: #f8f9fa; color: #333; padding: 2rem; }"
    html += ".container { max-width: 1200px; margin: 0 auto; }"
    html += "h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }"
    html += ".subtitle { color: #666; margin-bottom: 2rem; font-size: 0.9rem; }"
    html += ".card { background: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }"
    html += ".card h2 { font-size: 1.2rem; margin-bottom: 1rem; border-bottom: 2px solid #4e79a7; padding-bottom: 0.5rem; }"
    html += "table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }"
    html += "th, td { padding: 0.6rem 0.8rem; text-align: center; border: 1px solid #e9ecef; }"
    html += "th { background: #f1f3f5; font-weight: 600; }"
    html += "th:first-child, td:first-child { text-align: left; }"
    html += ".chart-container { position: relative; height: 400px; margin: 1rem 0; }"
    html += "footer { text-align: center; margin-top: 2rem; color: #999; font-size: 0.8rem; }"
    html += "</style></head><body><div class='container'>"

    html += "<h1>Korean LLM Benchmark Report</h1>"
    html += "<p class='subtitle'>Generated: " + timestamp + " | Model: " + model_name + " | Tasks: " + str(len(tasks_sorted)) + "</p>"

    html += '<div class="card"><h2>Overall Accuracy</h2>'
    html += '<p style="font-size:2rem; font-weight:700; text-align:center;">' + str(overall_avg) + "%</p>"
    html += '<p style="text-align:center; color:#666;">' + model_name + "</p></div>"

    if tasks_sorted:
        html += '<div class="card"><h2>Score Comparison</h2>'
        html += '<div class="chart-container"><canvas id="mainChart"></canvas></div></div>'

    html += '<div class="card"><h2>Detailed Results</h2><table>'
    html += "<thead><tr><th>Benchmark</th><th>Accuracy (%)</th></tr></thead><tbody>"
    for bm_id in tasks_sorted:
        html += "<tr><td>" + bm_id + "</td><td>" + str(score_table[bm_id]) + "</td></tr>"
    html += '<tr style="font-weight:700; background:#e9ecef;"><td>Overall</td><td>' + str(overall_avg) + "</td></tr>"
    html += "</tbody></table></div>"

    if perf_metrics:
        html += '<div class="card"><h2>Performance</h2><table>'
        html += "<thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"
        for k in ["output_tokens_per_second", "prompt_tokens_per_second", "requests_per_second"]:
            v = perf_metrics.get(k)
            if v is not None:
                html += "<tr><td>" + k.replace("_", " ").title() + "</td><td>" + str(round(float(v), 2)) + "</td></tr>"
        html += "</tbody></table></div>"

    if tasks_sorted:
        data_str = ", ".join(str(score_table[t]) for t in tasks_sorted)
        labels_str = ", ".join('"' + t + '"' for t in tasks_sorted)
        html += "<script>"
        html += "new Chart(document.getElementById('mainChart'), {"
        html += "type: 'bar', data: {"
        html += "labels: [" + labels_str + "],"
        html += 'datasets: [{ label: "' + model_name + '", data: [' + data_str + '],'
        html += 'backgroundColor: "#4e79a780", borderColor: "#4e79a7", borderWidth: 1 }]'
        html += "}, options: { responsive: true, maintainAspectRatio: false,"
        html += "scales: { y: { beginAtZero: true, max: 100,"
        html += "title: { display: true, text: 'Accuracy (%)' } } } } });"
        html += "</script>"

    html += "<footer>Generated by Korean LLM Evaluation Pipeline</footer>"
    html += "</div></body></html>"

    with open(report_html.path, "w") as f:
        f.write(html)
    print("report.html generated")


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Send Notification
# ──────────────────────────────────────────────────────────────────────────────
@dsl.component(base_image=BASE_IMAGE, packages_to_install=["requests"])
def send_notification(
    accuracy_results: Input[Dataset],
    performance_results: Input[Dataset],
    slack_webhook_url: str,
):
    """Send evaluation summary via Slack Incoming Webhook."""
    import json

    import requests

    acc_data = json.load(open(accuracy_results.path))
    perf_data = json.load(open(performance_results.path))

    model = acc_data.get("model", "unknown")
    benchmarks = acc_data.get("benchmarks", {})
    perf_metrics = perf_data.get("metrics", {})

    lines = []
    lines.append(":white_check_mark: *" + model + "* evaluation complete")
    lines.append("")
    lines.append("*Accuracy:*")
    for bm_id in sorted(benchmarks.keys()):
        metrics = benchmarks[bm_id].get("metrics", {})
        acc = metrics.get("overall_accuracy", "N/A")
        state = benchmarks[bm_id].get("state", "unknown")
        icon = ":large_green_circle:" if state == "completed" else ":red_circle:"
        lines.append("  " + icon + " " + bm_id + ": " + str(acc) + "%")

    if perf_metrics:
        lines.append("")
        lines.append("*Performance:*")
        out_tps = perf_metrics.get("output_tokens_per_second")
        if out_tps is not None:
            lines.append("  Output: " + str(round(float(out_tps), 2)) + " tok/s")
        prompt_tps = perf_metrics.get("prompt_tokens_per_second")
        if prompt_tps is not None:
            lines.append("  Prompt: " + str(round(float(prompt_tps), 2)) + " tok/s")

    msg = "\n".join(lines)
    print("Notification message:")
    print(msg)

    if slack_webhook_url:
        resp = requests.post(
            slack_webhook_url,
            json={"text": msg},
            timeout=10,
        )
        print("Slack response: " + str(resp.status_code))
    else:
        print("No Slack webhook URL provided. Skipping notification.")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline DAG
# ──────────────────────────────────────────────────────────────────────────────
@dsl.pipeline(
    name="korean-llm-eval-pipeline",
    description="Korean LLM evaluation: accuracy (5 benchmarks) + GuideLLM performance + report + Slack notification",
)
def eval_pipeline(
    evalhub_url: str,
    auth_token: str,
    model_endpoint: str,
    namespace: str = "demo",
    model_name: str = "glm-53-flash",
    model_api_key: str = "",
    provider_name: str = "Korean MCQ Evaluation",
    experiment_name: str = "pipeline-eval",
    limit: int = 10000,
    slack_webhook_url: str = "",
):
    """End-to-end Korean LLM evaluation pipeline.

    Args:
        evalhub_url: EvalHub API base URL
        auth_token: OpenShift auth token (oc whoami -t)
        model_endpoint: Model inference endpoint URL
        namespace: OpenShift namespace (X-Tenant for EvalHub API)
        model_name: Display name for the model
        model_api_key: K8s Secret name containing model API key (e.g. "model-api-key")
        provider_name: EvalHub provider name for Korean MCQ benchmarks
        experiment_name: MLflow experiment name for this run
        limit: Max samples per benchmark (0 = all)
        slack_webhook_url: Slack Incoming Webhook URL (optional)
    """
    # Step 1: Setup
    setup_task = setup_and_verify(
        evalhub_url=evalhub_url,
        auth_token=auth_token,
        namespace=namespace,
        provider_name=provider_name,
    )

    # Step 2: Accuracy (runs after setup)
    accuracy_task = run_accuracy_benchmarks(
        evalhub_url=evalhub_url,
        auth_token=auth_token,
        namespace=namespace,
        model_endpoint=model_endpoint,
        model_name=model_name,
        model_api_key=model_api_key,
        provider_id=setup_task.output,
        limit=limit,
        experiment_name=experiment_name,
    )

    # Step 3: Performance (runs after setup, parallel with accuracy)
    perf_task = run_performance_benchmark(
        evalhub_url=evalhub_url,
        auth_token=auth_token,
        namespace=namespace,
        model_endpoint=model_endpoint,
        model_name=model_name,
        experiment_name=experiment_name,
    )
    perf_task.after(setup_task)

    # Step 4: Reports (after both accuracy and performance)
    report_task = generate_reports(
        accuracy_results=accuracy_task.outputs["results"],
        performance_results=perf_task.outputs["results"],
    )

    # Step 5: Notify (after reports)
    send_notification(
        accuracy_results=accuracy_task.outputs["results"],
        performance_results=perf_task.outputs["results"],
        slack_webhook_url=slack_webhook_url,
    ).after(report_task)


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=eval_pipeline,
        package_path="eval_pipeline.yaml",
    )
    print("Pipeline compiled to eval_pipeline.yaml")
