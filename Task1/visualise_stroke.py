"""Task 1: Dataset selection and visualisation for healthcare analytics.

Uses only Python standard library so it can run in restricted environments.
Creates lightweight SVG charts in Task1/outputs/.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

DATASET_PATH = Path("healthcare-dataset-stroke-data.csv")
OUTPUT_DIR = Path("Task1/outputs")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def stroke_rate(rows: list[dict[str, str]]) -> float:
    stroke_cases = sum(int(r["stroke"]) for r in rows)
    return stroke_cases / len(rows)


def save_text_summary(rows: list[dict[str, str]]) -> None:
    ages_stroke = [float(r["age"]) for r in rows if r["stroke"] == "1" and r["age"]]
    ages_no_stroke = [float(r["age"]) for r in rows if r["stroke"] == "0" and r["age"]]
    bmi_missing = sum(1 for r in rows if not r["bmi"])

    lines = [
        f"Rows: {len(rows)}",
        f"Columns: {len(rows[0]) if rows else 0}",
        f"Stroke prevalence: {stroke_rate(rows) * 100:.2f}%",
        f"Mean age (stroke=1): {mean(ages_stroke):.2f}",
        f"Mean age (stroke=0): {mean(ages_no_stroke):.2f}",
        f"Missing BMI values: {bmi_missing}",
    ]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<text x="20" y="28" font-family="Arial" font-size="18" font-weight="bold">{title}</text>',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
    ]


def save_bar_chart_hypertension(rows: list[dict[str, str]]) -> None:
    grouped = defaultdict(lambda: Counter())
    for r in rows:
        grouped[r["hypertension"]][r["stroke"]] += 1

    categories = ["0", "1"]
    width, height = 760, 460
    left, top, chart_w, chart_h = 80, 70, 620, 320

    max_count = max(grouped[c][s] for c in categories for s in ["0", "1"])

    lines = svg_header(width, height, "Stroke counts by hypertension status")
    lines += [
        f'<line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="black"/>',
    ]

    colors = {"0": "#4e79a7", "1": "#e15759"}
    labels = {"0": "No stroke", "1": "Stroke"}
    group_w = chart_w / len(categories)
    bar_w = group_w * 0.28

    for i, cat in enumerate(categories):
        gx = left + i * group_w + group_w * 0.2
        lines.append(f'<text x="{gx+bar_w}" y="{top+chart_h+20}" font-size="12">Hypertension={cat}</text>')
        for j, stroke in enumerate(["0", "1"]):
            count = grouped[cat][stroke]
            h = (count / max_count) * (chart_h - 10)
            x = gx + j * (bar_w + 12)
            y = top + chart_h - h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors[stroke]}"/>')
            lines.append(f'<text x="{x:.1f}" y="{y-6:.1f}" font-size="11">{count}</text>')

    lines += [
        '<rect x="500" y="80" width="12" height="12" fill="#4e79a7"/><text x="518" y="90" font-size="12">No stroke</text>',
        '<rect x="500" y="100" width="12" height="12" fill="#e15759"/><text x="518" y="110" font-size="12">Stroke</text>',
        "</svg>",
    ]
    (OUTPUT_DIR / "bar_hypertension_stroke.svg").write_text("\n".join(lines), encoding="utf-8")


def save_age_histogram(rows: list[dict[str, str]]) -> None:
    bins = [(i, i + 10) for i in range(0, 100, 10)]
    counts = Counter()
    for r in rows:
        age_str = r.get("age", "")
        if not age_str:
            continue
        age = float(age_str)
        lo = int(age // 10) * 10
        lo = min(max(lo, 0), 90)
        counts[lo] += 1

    width, height = 760, 460
    left, top, chart_w, chart_h = 80, 70, 620, 320
    max_count = max(counts.values())

    lines = svg_header(width, height, "Age distribution (10-year bins)")
    lines += [
        f'<line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="black"/>',
    ]

    bar_w = chart_w / len(bins) * 0.75
    for i, (lo, hi) in enumerate(bins):
        count = counts.get(lo, 0)
        h = (count / max_count) * (chart_h - 10)
        x = left + i * (chart_w / len(bins)) + 8
        y = top + chart_h - h
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#59a14f"/>')
        lines.append(f'<text x="{x:.1f}" y="{top+chart_h+18}" font-size="11">{lo}-{hi-1}</text>')

    lines.append("</svg>")
    (OUTPUT_DIR / "hist_age.svg").write_text("\n".join(lines), encoding="utf-8")


def save_scatter_age_glucose(rows: list[dict[str, str]]) -> None:
    points = []
    for r in rows:
        try:
            age = float(r["age"])
            glucose = float(r["avg_glucose_level"])
        except (ValueError, KeyError):
            continue
        points.append((age, glucose, r["stroke"]))

    min_age = min(p[0] for p in points)
    max_age = max(p[0] for p in points)
    min_g = min(p[1] for p in points)
    max_g = max(p[1] for p in points)

    width, height = 760, 460
    left, top, chart_w, chart_h = 80, 70, 620, 320

    def x_scale(v: float) -> float:
        return left + ((v - min_age) / (max_age - min_age)) * chart_w

    def y_scale(v: float) -> float:
        return top + chart_h - ((v - min_g) / (max_g - min_g)) * chart_h

    lines = svg_header(width, height, "Age vs avg glucose level (colored by stroke)")
    lines += [
        f'<line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="black"/>',
    ]

    for age, gluc, stroke in points:
        color = "#e15759" if stroke == "1" else "#4e79a7"
        lines.append(
            f'<circle cx="{x_scale(age):.1f}" cy="{y_scale(gluc):.1f}" r="2" fill="{color}" fill-opacity="0.45"/>'
        )

    lines += [
        '<rect x="500" y="80" width="12" height="12" fill="#4e79a7"/><text x="518" y="90" font-size="12">No stroke</text>',
        '<rect x="500" y="100" width="12" height="12" fill="#e15759"/><text x="518" y="110" font-size="12">Stroke</text>',
        "</svg>",
    ]

    (OUTPUT_DIR / "scatter_age_glucose_stroke.svg").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(DATASET_PATH)
    save_text_summary(rows)
    save_bar_chart_hypertension(rows)
    save_age_histogram(rows)
    save_scatter_age_glucose(rows)
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
