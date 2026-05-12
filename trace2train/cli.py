"""ApexTrain-Core CLI — trace-to-train pipeline."""

import argparse
import yaml
from pathlib import Path

from trace2train.trace.loader import load_traces, group_by_request
from trace2train.trace.validator import validate_trace
from trace2train.dataset.sft_builder import build_sft_samples
from trace2train.dataset.dpo_builder import build_dpo_pairs
from trace2train.dataset.eval_builder import build_eval_samples
from trace2train.dataset.report import generate_report
from trace2train.train.tiny_trainer import TinyTrainer
from trace2train.eval.harness import run_evaluation
from trace2train.eval.metrics import compute_metrics, format_metrics_table


def cmd_extract(args):
    events = load_traces(args.traces)
    traces = group_by_request(events)
    build_sft_samples(traces, args.out)


def cmd_build_dpo(args):
    events = load_traces(args.traces)
    traces = group_by_request(events)
    build_dpo_pairs(traces, args.out)


def cmd_build_eval(args):
    events = load_traces(args.traces)
    traces = group_by_request(events)
    build_eval_samples(traces, args.out)


def cmd_report(args):
    generate_report(
        args.sft if hasattr(args, "sft") else "data/sft.jsonl",
        args.dpo if hasattr(args, "dpo") else "data/dpo.jsonl",
        args.eval if hasattr(args, "eval") else "data/eval.jsonl",
        args.out,
    )


def cmd_train(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    trainer = TinyTrainer(config)
    trainer.train()


def cmd_eval(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    ckpt = config.get("output", {}).get("checkpoint_dir", "checkpoints/") + "checkpoint.json"
    eval_data = config.get("data", {}).get("eval", "data/eval.jsonl")
    results = run_evaluation(ckpt, eval_data)
    metrics = compute_metrics(results)
    print(format_metrics_table(metrics))


def main():
    parser = argparse.ArgumentParser(description="ApexTrain-Core")
    sub = parser.add_subparsers(dest="command")

    p_extract = sub.add_parser("extract")
    p_extract.add_argument("traces", nargs="+")
    p_extract.add_argument("--out", default="data/sft.jsonl")

    p_dpo = sub.add_parser("build-dpo")
    p_dpo.add_argument("traces", nargs="+")
    p_dpo.add_argument("--out", default="data/dpo.jsonl")

    p_eval = sub.add_parser("build-eval")
    p_eval.add_argument("traces", nargs="+")
    p_eval.add_argument("--out", default="data/eval.jsonl")

    p_report = sub.add_parser("report")
    p_report.add_argument("--sft", default="data/sft.jsonl")
    p_report.add_argument("--dpo", default="data/dpo.jsonl")
    p_report.add_argument("--eval", default="data/eval.jsonl")
    p_report.add_argument("--out", default="reports/dataset.md")

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", default="examples/configs/tiny_sft.yaml")

    p_eval_cmd = sub.add_parser("eval")
    p_eval_cmd.add_argument("--config", default="examples/configs/eval.yaml")

    args = parser.parse_args()
    cmds = {
        "extract": cmd_extract,
        "build-dpo": cmd_build_dpo,
        "build-eval": cmd_build_eval,
        "report": cmd_report,
        "train": cmd_train,
        "eval": cmd_eval,
    }

    if args.command in cmds:
        cmds[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
