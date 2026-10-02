#!/usr/bin/env python3
"""
Compile the evaluation pipeline to YAML.

Usage:
    python compile.py
    python compile.py --output my_pipeline.yaml

The generated YAML can be imported via:
    RHOAI Dashboard → Pipelines → Import pipeline → Upload file
"""

import argparse

from kfp import compiler
from pipeline import eval_pipeline


def main():
    parser = argparse.ArgumentParser(description="Compile Korean LLM eval pipeline to YAML")
    parser.add_argument(
        "--output",
        default="eval_pipeline.yaml",
        help="Output YAML file path (default: eval_pipeline.yaml)",
    )
    args = parser.parse_args()

    compiler.Compiler().compile(
        pipeline_func=eval_pipeline,
        package_path=args.output,
    )
    print(f"Pipeline compiled: {args.output}")
    print()
    print("Next steps:")
    print("  1. RHOAI Dashboard → Pipelines → Import pipeline")
    print(f"  2. Upload {args.output}")
    print("  3. Create run → fill in parameters → Create")


if __name__ == "__main__":
    main()
