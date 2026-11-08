
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREND_MAP = {"positive": 0, "neutral": 1, "negative": 2}
DELIMITER = "——"
COLUMNS = ["id", "label", "L0_news_id", "L0_news_trend",
           "L1_comments_id", "L1_L0_s", "L1_comments_trend",
           "L2_comments_id", "L2_L0_s", "L2_comments_trend"]


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def index_rows(rows, key, name):
    result = {}
    for row in rows:
        identity = row[key]
        if not identity.isdigit() or identity in result:
            raise ValueError(f"{name}: missing, nonnumeric or duplicate {key}: {identity}")
        result[identity] = row
    return result


def sentiment(row, required=False):
    value = row["sentiment_trend"].strip().lower()
    if value in TREND_MAP:
        return str(TREND_MAP[value])
    if value not in ("", "error", "none", "nan") or required:
        raise ValueError(f"Invalid/missing sentiment for {row.get('node_id', row['news_id'])}")
    return ""


def timestamp(row):
    value = datetime.fromisoformat(row["created_at_utc"].replace("Z", "+00:00"))
    if value.tzinfo is None:
        # The source CSV explicitly declares UTC in its column name, but
        # serializes some values without an offset. Never use machine time.
        value = value.replace(tzinfo=timezone.utc)
    return value


def delay(row, news):
    seconds = (timestamp(row) - timestamp(news)).total_seconds()
    if seconds < 0 or not seconds.is_integer():
        raise ValueError(f"Expected nonnegative whole-second delay for {row['node_id']}")
    if row.get("delay_seconds", "") and float(row["delay_seconds"]) != seconds:
        raise ValueError(f"Stored delay differs from timestamps for {row['node_id']}")
    return str(int(seconds))


def build_records(news_rows, l1_rows, l2_rows):
    news = index_rows(news_rows, "news_id", "news")
    l1 = index_rows(l1_rows, "node_id", "L1")
    l2 = index_rows(l2_rows, "node_id", "L2")
    children, replies = defaultdict(list), defaultdict(list)
    missing = []
    for row in news_rows:
        if row["label"] not in ("real", "fake"):
            raise ValueError("Only real/fake news is supported")
        sentiment(row, required=True)
    for depth, rows in [("1", l1_rows), ("2", l2_rows)]:
        for row in rows:
            root = news.get(row["news_id"])
            if root is None or row["depth"] != depth:
                raise ValueError(f"Invalid root/depth for {row['node_id']}")
            if row.get("event", root.get("event")) != root.get("event"):
                raise ValueError("Event mismatch")
            if row.get("label", root["label"]) != root["label"]:
                raise ValueError("News label mismatch")
            delay(row, root)
            if not sentiment(row):
                missing.append({"depth": depth, "news_id": row["news_id"], "node_id": row["node_id"]})
            if depth == "1":
                if row["parent_id"] != row["news_id"]:
                    raise ValueError("L1 parent is not the root news")
                children[row["news_id"]].append(row)
            else:
                parent = l1.get(row["parent_id"])
                if parent is None or parent["news_id"] != row["news_id"]:
                    raise ValueError("Missing or cross-news L2 parent")
                if timestamp(row) < timestamp(parent):
                    raise ValueError("L2 precedes its parent")
                replies[row["parent_id"]].append(row)
    records = []
    for root in news_rows:
        for comment in children[root["news_id"]] or [None]:
            descendants = replies[comment["node_id"]] if comment else []
            for reply in descendants or [None]:
                records.append({
                    "id": str(len(records)), "label": root["label"],
                    "L0_news_id": root["news_id"], "L0_news_trend": sentiment(root, True),
                    "L1_comments_id": comment["node_id"] if comment else "",
                    "L1_L0_s": delay(comment, root) if comment else "",
                    "L1_comments_trend": sentiment(comment) if comment else "",
                    "L2_comments_id": reply["node_id"] if reply else "",
                    "L2_L0_s": delay(reply, root) if reply else "",
                    "L2_comments_trend": sentiment(reply) if reply else "",
                })
    return records, missing


def serialize(row):
    n = f"N_{row['L0_news_id']}_{row['L0_news_trend']}"
    u = f"C1_{row['L1_comments_id'] or 'None'}_{row['L1_comments_trend'] or 'None'}"
    parts = [n, row["L1_L0_s"] or "0", u]
    # Match build_metapaths.py: missing L2 sentiment falls back to NU.
    if all(row[k] != "" for k in ["L2_comments_id", "L2_comments_trend", "L2_L0_s"]):
        parts.extend([row["L2_L0_s"], f"C2_{row['L2_comments_id']}_{row['L2_comments_trend']}"])
    return DELIMITER.join(parts)


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentiment-dir", type=Path, default=Path("sentiment_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("metapath_data"))
    parser.add_argument("--check-only", action="store_true", help="Validate without writing outputs")
    parser.add_argument("--strict-sentiment", action="store_true", help="Fail on missing L1/L2 sentiment")
    args = parser.parse_args()
    source = args.sentiment_dir if args.sentiment_dir.is_absolute() else HERE / args.sentiment_dir
    output = args.output_dir if args.output_dir.is_absolute() else HERE / args.output_dir
    inputs = [source / f"pheme_{level}_sentiment.csv" for level in ("news", "L1", "L2")]
    tables = [read_table(path) for path in inputs]
    records, missing = build_records(*tables)
    if args.strict_sentiment and missing:
        raise ValueError(f"{len(missing)} nodes lack sentiment; complete annotation first")
    summary = {
        "sentiment_mapping": TREND_MAP,
        "input_counts": dict(zip(("news", "L1", "L2"), map(len, tables))),
        "news_labels": dict(Counter(r["label"] for r in tables[0])),
        "path_rows": len(records),
        "nu_rows": sum(len(serialize(r).split(DELIMITER)) == 3 and bool(r["L1_comments_id"]) for r in records),
        "nuv_rows": sum(len(serialize(r).split(DELIMITER)) == 5 for r in records),
        "placeholder_rows": sum(not r["L1_comments_id"] for r in records),
        "missing_sentiment_nodes": missing,
        "missing_l1_path_rows": sum(bool(r["L1_comments_id"]) and not r["L1_comments_trend"] for r in records),
        "l2_missing_fallback_rows": sum(bool(r["L2_comments_id"]) and not r["L2_comments_trend"] for r in records),
        "time_unit": "whole seconds from news publication; both L1 and L2 offsets retained",
        "path_policy": "NUV per reply; NU only when no reply (or L2 sentiment missing); placeholder for no L1",
        "id_policy": "Original numeric node IDs preserved as strings; news order follows input table",
        "split_policy": "No train/validation/test split created",
        "inputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
    }
    print(json.dumps({k: summary[k] for k in ["input_counts", "news_labels", "path_rows", "nu_rows", "nuv_rows", "placeholder_rows", "missing_l1_path_rows"]}, indent=2))
    if args.check_only:
        return
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for label in ("real", "fake"):
        rows = [r for r in records if r["label"] == label]
        encoded = output / f"pheme_{label}_id.csv"
        write_csv(encoded, rows, COLUMNS)
        paths = output / f"pheme_{label}_metapaths.txt"
        paths.write_text("".join(serialize(r) + "\n" for r in rows), encoding="utf-8")
        ids = output / f"pheme_{label}_news_ids.csv"
        identities = [{"row_index": str(i), "news_id": r["news_id"], "label": label}
                      for i, r in enumerate(n for n in tables[0] if n["label"] == label)]
        write_csv(ids, identities, ["row_index", "news_id", "label"])
        files.extend([encoded, paths, ids])
    summary["outputs"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metapaths to {output}")


if __name__ == "__main__":
    main()
