"""Dependency-free readers for lm-eval results (raw or wrapped by eval.py)."""

import math
import re


RESULT_COLUMNS = ["dataset", "metric", "filter", "context_length", "value", "stderr", "result_type"]


def finite_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def context_length(task, metric):
    """Dataset/metric length, never infer actual input length from a model's cap."""
    if metric.isdigit():
        return int(metric)
    match = re.search(r"_(\d+)k$", task)
    if match:
        return int(match[1]) * 1024
    return None


def result_rows(payload):
    """Keep every numeric metric and filter, plus group aggregates and stderr."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise ValueError("Expected an lm-eval JSON object with a results mapping")
    report = payload
    if isinstance(report["results"].get("results"), dict):
        report = report["results"]  # eval.py envelope
    rows = []
    seen = set()
    for section in ("results", "groups"):
        for task, metrics in report.get(section, {}).items():
            if not isinstance(metrics, dict):
                continue
            for key, value in metrics.items():
                metric, _, filter_name = key.partition(",")
                if metric.endswith("_stderr") or metric in ("alias", "samples"):
                    continue
                number = finite_number(value)
                if number is None:
                    continue
                filter_name = filter_name or "none"
                identity = (task, metric, filter_name)
                if identity in seen:  # group summaries also appear in results in some harness versions
                    continue
                seen.add(identity)
                stderr = metrics.get(f"{metric}_stderr,{filter_name}", metrics.get(f"{metric}_stderr"))
                rows.append(dict(dataset=task, metric=metric, filter=filter_name,
                                 context_length=context_length(task, metric), value=number,
                                 stderr=finite_number(stderr),
                                 result_type="group" if section == "groups" or task in report.get("groups", {}) else "task"))
    return rows


def markdown_rows(text):
    """Read make_table output, including multiple tables embedded in console logs."""
    header, task = None, None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if "Tasks" in cells and "Metric" in cells and "Value" in cells:
            header, task = cells, None
            continue
        if header is None or len(cells) != len(header):
            continue
        item = dict(zip(header, cells))
        value = finite_number(item["Value"])
        if value is None:
            continue
        task = re.sub(r"^[- ]+", "", item["Tasks"]) or task
        if not task:
            continue
        metric = item["Metric"]
        yield dict(dataset=task, metric=metric, filter=item.get("Filter") or "none",
                   context_length=context_length(task, metric), value=value,
                   stderr=finite_number(item.get("Stderr")), result_type="task")
