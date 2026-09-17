"""Create a read-only, file-by-file profile of the Petrobras 3W dataset."""

from __future__ import annotations

import configparser
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "3W" / "dataset"
REPORT = ROOT / "reports" / "dataset_profile.txt"
LABELS = {0: "Normal Operation", 1: "Abrupt Increase of BSW", 2: "Spurious Closure of DHSV", 3: "Severe Slugging", 4: "Flow Instability", 5: "Rapid Productivity Loss", 6: "Quick Restriction in PCK", 7: "Scaling in PCK", 8: "Hydrate in Production Line", 9: "Hydrate in Service Line"}


def percent(value: float, total: int) -> str:
    """Format a percentage safely, including when a category has no rows."""
    return f"{100 * value / total:.4f}%" if total else "0.0000%"


def value_counts_text(series: pd.Series) -> str:
    """Show frequencies while retaining missing values for honest profiling."""
    return ", ".join(f"{repr(k)}={v}" for k, v in series.value_counts(dropna=False).to_dict().items()) or "none"


def source_id(filename: str) -> str:
    """Extract only the well-like prefix actually present in a filename."""
    stem = Path(filename).stem
    match = re.match(r"(WELL[-_]?[A-Za-z0-9]+)", stem, flags=re.IGNORECASE)
    return match.group(1) if match else stem.split("_")[0]


def quantiles(values: list[float]) -> str:
    """Summarize per-instance measurements with useful robust percentiles."""
    series = pd.Series(values)
    q = series.quantile([.01, .25, .50, .75, .99])
    return (f"min={series.min():.3f}, p01={q[.01]:.3f}, p25={q[.25]:.3f}, "
            f"median={q[.50]:.3f}, mean={series.mean():.3f}, p75={q[.75]:.3f}, "
            f"p99={q[.99]:.3f}, max={series.max():.3f}")


def main() -> None:
    """Scan each Parquet file once and write measured results and interpretations."""
    ini = configparser.ConfigParser()
    # Preserve the dataset's case-sensitive column names for exact schema checks.
    ini.optionxform = str
    ini.read(DATASET / "dataset.ini", encoding="utf-8")
    sensor_columns = [k for k in ini["PARQUET_FILE_PROPERTIES"] if k not in {"timestamp", "class", "state"}]
    paths = [p for label in range(10) for p in sorted((DATASET / str(label)).glob("*.parquet"))]
    total_rows = 0
    rows_by_label, files_by_label, missing = Counter(), Counter(), Counter()
    sizes, durations = [], []
    source_files = defaultdict(list)
    schemas = defaultdict(list)
    representatives = {}

    for path in paths:
        # One-file-at-a-time processing keeps memory bounded for the large raw dataset.
        frame = pd.read_parquet(path, engine="pyarrow")
        label_dir = int(path.parent.name)
        timestamps = pd.to_datetime(frame.index)
        rows = len(frame)
        total_rows += rows
        files_by_label[label_dir] += 1
        rows_by_label[label_dir] += rows
        sizes.append(rows)
        durations.append((timestamps.max() - timestamps.min()).total_seconds() if rows else 0.0)
        missing.update(frame.reindex(columns=sensor_columns).isna().sum().to_dict())
        schemas[tuple(frame.columns)].append(str(path.relative_to(ROOT)))
        source_files[source_id(path.name)].append(str(path.relative_to(ROOT)))
        # Keeping ten small copies permits representative behavior checks after scanning.
        representatives.setdefault(label_dir, (path, frame.copy()))

    def section(title: str) -> list[str]:
        """Create consistent headings so the report is easy to navigate."""
        return ["", title, "-" * len(title)]

    expected = tuple(sensor_columns + ["class", "state"])
    lines = ["Petrobras 3W Dataset Profile", "=============================="]
    lines += section("Dataset overview") + [f"Dataset version: {ini['VERSION']['DATASET']}", f"Parquet files (instances): {len(paths)}", f"Observations (rows): {total_rows}", f"Columns excluding timestamp index: {len(expected)}", f"Expected structure confirmed: {len(sensor_columns)} sensor/operational variables + class + state", "Timestamp stored as DataFrame index: yes"]
    lines += section("Instance counts")
    for label in range(10):
        lines.append(f"{label}: {LABELS[label]} | {files_by_label[label]} files ({percent(files_by_label[label], len(paths))} of instances)")
    lines += section("Observation counts")
    for label in range(10):
        lines.append(f"{label}: {LABELS[label]} | {rows_by_label[label]} rows ({percent(rows_by_label[label], total_rows)} of observations)")
    lines += section("Instance size (rows per Parquet file)") + [quantiles([float(x) for x in sizes])]
    lines += section("Instance duration (seconds)") + [quantiles(durations)]
    lines += section("Missing sensor observations")
    for column in sensor_columns:
        lines.append(f"{column}: {missing[column]} ({percent(missing[column], total_rows)})")
    lines += section("Schema consistency")
    lines += [f"Schemas consistent: {len(schemas) == 1 and expected in schemas}", f"Expected columns: {list(expected)}"]
    for schema, names in schemas.items():
        if schema != expected:
            lines.append(f"Unexpected schema ({len(schema)} columns), files={names[:10]}")
    lines += section("Source/well information")
    repeated = {key: names for key, names in source_files.items() if len(names) > 1}
    lines += [f"Unique source/well identifiers: {len(source_files)}", f"Identifiers used by multiple instances: {len(repeated)}"]
    for key, names in sorted(repeated.items())[:20]:
        lines.append(f"{key}: {len(names)} instances; examples={names[:3]}")
    lines += section("Representative class, state, and temporal behavior")
    for label in range(10):
        path, frame = representatives[label]
        steps = pd.Series(pd.to_datetime(frame.index)).diff().dt.total_seconds().dropna()
        median_step = steps.median() if not steps.empty else float("nan")
        lines += [f"Class {label} ({LABELS[label]}): {path.relative_to(ROOT)}", f"  class unique/frequencies: {value_counts_text(frame['class'])}", f"  class changes within instance: {frame['class'].nunique(dropna=False) > 1}", f"  state unique/frequencies: {value_counts_text(frame['state'])}", f"  state changes within instance: {frame['state'].nunique(dropna=False) > 1}", f"  timestamps monotonically increasing: {frame.index.is_monotonic_increasing}", f"  median step seconds: {median_step}", f"  approximately one second apart: {pd.notna(median_step) and abs(median_step - 1) <= 0.1}"]
        if label > 0:
            # Missing labels cannot be event evidence, so resolve them before Boolean use.
            event_rows = (frame["class"] == label).fillna(False)
            positions = [i for i, present in enumerate(event_rows.tolist()) if present]
            first, last = (min(positions), max(positions)) if positions else (None, None)
            lines.append(f"  event-phase evidence: before={bool(first and first > 0)}, event={bool(positions)}, after={bool(last is not None and last < len(frame) - 1)}; measured from class labels")
    lines += section("Label / event phase information") + ["The representative section reports measured before/event/after evidence; no phase boundaries were invented."]
    lines += section("Potential modeling implications (interpretation of measured results)") + ["- Use instance and observation percentages above when assessing class imbalance.", "- Missingness varies by sensor according to the table and may need explicit handling.", "- Timestamp ordering and one-second spacing indicate temporal dependence should be respected.", "- Repeated source/well identifiers support grouped train/test splitting to reduce source leakage.", "- Label transitions should be inspected before defining an early-warning target; this profile does not choose one.", "- No cost values, preprocessing, feature engineering, or models were used."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
