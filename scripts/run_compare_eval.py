import argparse
import ast
import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE_EVAL_DIR = Path("output/model_eval_base")
RL_EVAL_DIR = Path("output/model_eval_rl")


def run_eval(
    repo_root: Path,
    model_name: str,
    dataset: str,
    template: str,
    dataset_name: str,
    trained_agent_path: Optional[str],
    outdir: Path,
    n_rows: int,
) -> None:
    """Run one evaluation (base or RL) via evaluate_model.py."""

    from subprocess import check_call

    args = [
        sys.executable,
        "-m",
        "sql_rl_gen.generation.evaluate_model",
        "--model_name_or_path",
        model_name,
        "--dataset",
        dataset,
        "--template",
        template,
        "--outdir",
        str(outdir),
        "--dataset_name",
        dataset_name,
        "--number_of_rows_to_use",
        str(n_rows),
    ]
    if trained_agent_path is not None:
        args.extend(["--trained_agent_path", trained_agent_path])

    print("Running:", " ".join(args))
    check_call(args, cwd=str(repo_root))


def load_eval_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load evaluate_model.py's statistics_metrics file into per-sample rows."""

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_in_file = list(reader)

    if not rows_in_file:
        return []

    # Usually the file has 1-2 lines; the last line contains per-sample lists.
    row = rows_in_file[-1]

    def parse_list(value: str) -> list:
        try:
            return ast.literal_eval(value)
        except Exception:
            return []

    accuracies = parse_list(row.get("accuracy", "[]"))
    outputs = parse_list(row.get("output", "[]"))
    expecteds = parse_list(row.get("expected", "[]"))

    n = min(len(accuracies), len(outputs), len(expecteds))

    per_sample: List[Dict[str, Any]] = []
    for i in range(n):
        out_i = outputs[i]
        # `output` can be a tuple like ("SQL ...",)
        if isinstance(out_i, tuple) and out_i:
            out_str = out_i[0]
        else:
            out_str = str(out_i)

        per_sample.append(
            {
                "accuracy": float(accuracies[i]),
                "output": out_str,
                "expected": str(expecteds[i]),
            }
        )

    return per_sample


def build_comparison_table(base_rows: List[Dict[str, Any]], rl_rows: List[Dict[str, Any]]) -> str:
    """Align base vs RL by row index and return a Markdown table."""

    lines: List[str] = []
    lines.append("## Sample-by-sample comparison\n")
    lines.append("| # | gold SQL (expected) | base SQL (output) | base acc | RL SQL (output) | RL acc |")
    lines.append("| - | ------------------- | ---------------- | -------- | --------------- | ------ |")

    n = min(len(base_rows), len(rl_rows))
    for idx in range(n):
        b = base_rows[idx]
        r = rl_rows[idx]
        expected = (b.get("expected") or r.get("expected") or "").replace("|", "\\|")
        base_out = (b.get("output") or "").replace("|", "\\|")
        rl_out = (r.get("output") or "").replace("|", "\\|")
        base_acc = b.get("accuracy")
        rl_acc = r.get("accuracy")
        lines.append(
            f"| {idx + 1} | `{expected}` | `{base_out}` | {base_acc} | `{rl_out}` | {rl_acc} |"
        )

    return "\n".join(lines) + "\n"


def _normalize_sql(text: str) -> str:
    """Coarse normalization to avoid counting pure formatting diffs as changes."""

    s = (text or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(";").strip()
    return s.lower()


def compute_summary(base_rows: List[Dict[str, Any]], rl_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = min(len(base_rows), len(rl_rows))
    if n == 0:
        return {
            "n_samples": 0,
            "base_acc_mean": 0.0,
            "rl_acc_mean": 0.0,
            "diff_count": 0,
            "changed_sql_count": 0,
        }

    base_acc_mean = sum(float(base_rows[i].get("accuracy", 0.0) or 0.0) for i in range(n)) / n
    rl_acc_mean = sum(float(rl_rows[i].get("accuracy", 0.0) or 0.0) for i in range(n)) / n
    diff_count = sum(
        1
        for i in range(n)
        if float(base_rows[i].get("accuracy", 0.0) or 0.0)
        != float(rl_rows[i].get("accuracy", 0.0) or 0.0)
    )
    changed_sql_count = sum(
        1
        for i in range(n)
        if _normalize_sql(str(base_rows[i].get("output") or ""))
        != _normalize_sql(str(rl_rows[i].get("output") or ""))
    )

    return {
        "n_samples": n,
        "base_acc_mean": base_acc_mean,
        "rl_acc_mean": rl_acc_mean,
        "diff_count": diff_count,
        "changed_sql_count": changed_sql_count,
    }


def get_git_info(repo_root: Path) -> Dict[str, str]:
    """Return git branch/commit info (best-effort)."""

    from subprocess import check_output

    def _run(cmd: List[str]) -> Optional[str]:
        try:
            return check_output(cmd, cwd=repo_root, text=True).strip()
        except Exception:
            return None

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run(["git", "rev-parse", "HEAD"])
    short_commit = _run(["git", "rev-parse", "--short", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    info = {
        "git_branch": branch,
        "git_commit": commit,
        "git_short_commit": short_commit,
        "git_dirty": ("true" if dirty else "false") if dirty is not None else None,
    }
    return {k: v for k, v in info.items() if v is not None}


def build_markdown_report(
    base_rows: List[Dict[str, Any]],
    rl_rows: List[Dict[str, Any]],
    run_params: Dict[str, Any],
    summary: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Base model vs RL model comparison\n")

    lines.append("## Run parameters")
    for k in [
        "dataset",
        "dataset_name",
        "template",
        "number_of_rows_to_use",
        "model_name_or_path",
        "trained_agent_path",
        "base_eval_dir",
        "rl_eval_dir",
        "git_branch",
        "git_short_commit",
        "git_commit",
        "git_dirty",
    ]:
        if k in run_params and run_params[k] is not None:
            lines.append(f"- {k}: `{run_params[k]}`")
    lines.append("")

    lines.append("## Summary metrics")
    lines.append(f"- base_acc_mean: {summary['base_acc_mean']:.6f}")
    lines.append(f"- rl_acc_mean: {summary['rl_acc_mean']:.6f}")
    lines.append(f"- diff_count: {summary['diff_count']}")
    lines.append(f"- changed_sql_count: {summary['changed_sql_count']}")
    lines.append("")

    lines.append(build_comparison_table(base_rows, rl_rows))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run base and RL evaluation and generate a Markdown comparison report"
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="example_text2sql_spider_dev")
    parser.add_argument("--dataset_name", type=str, default="spider")
    parser.add_argument("--template", type=str, default="llama3")
    parser.add_argument("--trained_agent_path", type=str, required=True)
    parser.add_argument("--number_of_rows_to_use", type=int, default=200)
    parser.add_argument("--out_md", type=str, default="output/compare_base_vs_rl.md")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    print(
        "[compare-eval] dataset={} dataset_name={} n_rows={}".format(
            args.dataset, args.dataset_name, args.number_of_rows_to_use
        )
    )
    if args.dataset.startswith("example_"):
        print(
            "[compare-eval] NOTE: You are evaluating an example dataset. "
            "If base vs RL looks identical, consider using a harder/larger eval set."
        )

    base_eval_dir = root / BASE_EVAL_DIR
    base_eval_dir.mkdir(parents=True, exist_ok=True)
    run_eval(
        repo_root=root,
        model_name=args.model_name_or_path,
        dataset=args.dataset,
        template=args.template,
        dataset_name=args.dataset_name,
        trained_agent_path=None,
        outdir=base_eval_dir,
        n_rows=args.number_of_rows_to_use,
    )

    rl_eval_dir = root / RL_EVAL_DIR
    rl_eval_dir.mkdir(parents=True, exist_ok=True)
    run_eval(
        repo_root=root,
        model_name=args.model_name_or_path,
        dataset=args.dataset,
        template=args.template,
        dataset_name=args.dataset_name,
        trained_agent_path=args.trained_agent_path,
        outdir=rl_eval_dir,
        n_rows=args.number_of_rows_to_use,
    )

    base_csv = base_eval_dir / "statistics_metrics"
    rl_csv = rl_eval_dir / "statistics_metrics"

    if not base_csv.exists():
        raise FileNotFoundError("Base evaluation result not found: {}".format(base_csv))
    if not rl_csv.exists():
        raise FileNotFoundError("RL evaluation result not found: {}".format(rl_csv))

    base_rows = load_eval_csv(base_csv)
    rl_rows = load_eval_csv(rl_csv)

    summary = compute_summary(base_rows, rl_rows)
    print(
        "[compare-eval] summary:",
        "base_acc_mean={:.6f}".format(summary["base_acc_mean"]),
        "rl_acc_mean={:.6f}".format(summary["rl_acc_mean"]),
        "diff_count={}".format(summary["diff_count"]),
        "changed_sql_count={}".format(summary["changed_sql_count"]),
        "n_samples={}".format(summary["n_samples"]),
    )

    run_params: Dict[str, Any] = {
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "template": args.template,
        "number_of_rows_to_use": args.number_of_rows_to_use,
        "model_name_or_path": args.model_name_or_path,
        "trained_agent_path": args.trained_agent_path,
        "base_eval_dir": str(base_eval_dir.resolve()),
        "rl_eval_dir": str(rl_eval_dir.resolve()),
    }
    run_params.update(get_git_info(root))

    md = build_markdown_report(base_rows, rl_rows, run_params, summary)

    out_md_path = root / args.out_md
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text(md, encoding="utf-8")
    print("Markdown report generated:", out_md_path)


if __name__ == "__main__":
    main()
