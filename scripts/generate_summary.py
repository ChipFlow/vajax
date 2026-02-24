#!/usr/bin/env python3
"""Generate a summary page from CI benchmark and test artifacts.

Reads JSON benchmark summaries and JUnit XML test results from a directory,
and outputs a single index.md with aggregated results.

Usage:
    uv run python scripts/generate_summary.py \
        --artifacts-dir ./artifacts \
        --output ./site/index.md
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree


def parse_junit_xml(xml_path: Path) -> dict:
    """Parse a JUnit XML file and return summary stats."""
    tree = ElementTree.parse(xml_path)
    root = tree.getroot()

    # Handle both <testsuites> and <testsuite> as root
    if root.tag == "testsuites":
        suites = root.findall("testsuite")
    else:
        suites = [root]

    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    total_time = 0.0

    for suite in suites:
        total_tests += int(suite.get("tests", 0))
        total_failures += int(suite.get("failures", 0))
        total_errors += int(suite.get("errors", 0))
        total_skipped += int(suite.get("skipped", 0))
        total_time += float(suite.get("time", 0))

    passed = total_tests - total_failures - total_errors - total_skipped
    return {
        "tests": total_tests,
        "passed": passed,
        "failed": total_failures,
        "errors": total_errors,
        "skipped": total_skipped,
        "time_s": round(total_time, 1),
    }


def find_artifacts(artifacts_dir: Path) -> tuple[list[dict], list[dict], dict[str, dict]]:
    """Scan artifacts directory for benchmark JSONs and JUnit XMLs.

    Returns (cpu_benchmarks, gpu_benchmarks, test_suites).
    """
    cpu_benchmarks = []
    gpu_benchmarks = []
    test_suites = {}

    for path in sorted(artifacts_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, list) or not data:
            continue
        # Distinguish GPU vs CPU by backend field
        first = data[0]
        if first.get("backend") in ("gpu", "cuda"):
            gpu_benchmarks.extend(data)
        else:
            cpu_benchmarks.extend(data)

    for path in sorted(artifacts_dir.rglob("*.xml")):
        try:
            stats = parse_junit_xml(path)
        except (ElementTree.ParseError, OSError):
            continue
        # Derive suite name from filename (e.g. test-results-unit.xml -> unit)
        name = (
            path.stem.replace("test-results-", "")
            .replace("ngspice-results", "ngspice")
            .replace("xyce-results", "xyce")
        )
        test_suites[name] = stats

    return cpu_benchmarks, gpu_benchmarks, test_suites


def render_test_table(test_suites: dict[str, dict]) -> str:
    """Render test coverage table."""
    if not test_suites:
        return "_No test results available._\n"

    lines = [
        "| Suite | Passed | Failed | Errors | Skipped | Total | Time |",
        "|-------|--------|--------|--------|---------|-------|------|",
    ]
    for name, s in sorted(test_suites.items()):
        status = "PASS" if s["failed"] == 0 and s["errors"] == 0 else "FAIL"
        lines.append(
            f"| {name} | {s['passed']} | {s['failed']} | {s['errors']} "
            f"| {s['skipped']} | {s['tests']} | {s['time_s']}s | {status} |"
        )

    # Totals
    totals = {
        k: sum(s[k] for s in test_suites.values())
        for k in ["tests", "passed", "failed", "errors", "skipped"]
    }
    total_time = round(sum(s["time_s"] for s in test_suites.values()), 1)
    lines.append(
        f"| **Total** | **{totals['passed']}** | **{totals['failed']}** | **{totals['errors']}** "
        f"| **{totals['skipped']}** | **{totals['tests']}** | {total_time}s |"
    )
    return "\n".join(lines) + "\n"


def render_benchmark_table(benchmarks: list[dict], title: str) -> str:
    """Render a benchmark performance table."""
    if not benchmarks:
        return f"_No {title.lower()} benchmark data available._\n"

    lines = [
        f"### {title}\n",
        "| Benchmark | Steps | VA-JAX (ms/step) | VACASK (ms/step) | Ratio | Startup |",
        "|-----------|-------|---------------------|------------------|-------|---------|",
    ]
    for b in benchmarks:
        vacask = f"{b['vacask_ms_per_step']:.4f}" if b.get("vacask_ms_per_step") else "N/A"
        ratio = f"{b['ratio']:.2f}x" if b.get("ratio") else "N/A"
        startup = f"{b.get('startup_s', 0):.1f}s"
        lines.append(
            f"| {b['name']} | {b['steps']:,} | {b['jax_ms_per_step']:.4f} | {vacask} | {ratio} | {startup} |"
        )
    return "\n".join(lines) + "\n"


def generate_summary(
    artifacts_dir: Path,
    output_path: Path,
    commit_sha: str = "",
    repo_url: str = "https://github.com/ChipFlow/va-jax",
) -> None:
    """Generate the summary markdown page."""
    cpu_benchmarks, gpu_benchmarks, test_suites = find_artifacts(artifacts_dir)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections = [
        "# VA-JAX CI Summary\n",
        f"_Last updated: {now}_\n",
    ]

    if commit_sha:
        short_sha = commit_sha[:8]
        sections.append(f"_Commit: [{short_sha}]({repo_url}/commit/{commit_sha})_\n")

    # Test coverage
    sections.append("## Test Coverage\n")
    sections.append(render_test_table(test_suites))

    # CPU benchmarks
    sections.append("\n## Performance\n")
    sections.append(render_benchmark_table(cpu_benchmarks, "CPU Benchmarks"))

    # GPU benchmarks
    sections.append(render_benchmark_table(gpu_benchmarks, "GPU Benchmarks"))

    # Links
    sections.append("\n---\n")
    sections.append(f"[View workflows]({repo_url}/actions) | ")
    sections.append(f"[Repository]({repo_url})\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections))
    print(f"Summary written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate CI summary page")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Directory containing downloaded CI artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("site/index.md"),
        help="Output markdown path (default: site/index.md)",
    )
    parser.add_argument(
        "--commit-sha",
        type=str,
        default="",
        help="Git commit SHA for linking",
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        default="https://github.com/ChipFlow/va-jax",
        help="Repository URL",
    )
    args = parser.parse_args()

    if not args.artifacts_dir.exists():
        print(f"Error: artifacts directory not found: {args.artifacts_dir}", file=sys.stderr)
        sys.exit(1)

    generate_summary(args.artifacts_dir, args.output, args.commit_sha, args.repo_url)


if __name__ == "__main__":
    main()
