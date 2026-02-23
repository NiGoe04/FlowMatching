from __future__ import annotations

import os
from datetime import datetime


def create_experiment_report(experiment_results: list[dict], reports_dir: str) -> str:
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(reports_dir, f"experiment_report_{timestamp}.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Flow Matching + OT Mass Experiment Report\n")
        f.write("=" * 50 + "\n\n")

        for idx, result in enumerate(experiment_results, start=1):
            combo = result["combination"]
            metrics = result["metrics"]

            f.write(f"Experiment #{idx}\n")
            f.write("-" * 50 + "\n")
            f.write(f"scenario: {combo['scenario']}\n")
            f.write(f"dim: {combo['dim']}\n")
            f.write(f"ot_batch_size: {combo['ot_batch_size']}\n")
            f.write(f"ot_optimizer: {combo['ot_optimizer']}\n")
            f.write(f"epsilon: {combo['epsilon']}\n")
            f.write(f"model_path: {combo['model_path']}\n")
            f.write("metrics:\n")
            for metric_name, metric_value in metrics.items():
                f.write(f"  - {metric_name}: {metric_value}\n")
            f.write("\n")

    return report_path
