"""Recompute Weibo-25 statistics from source tables with explicit denominators.

Only nonempty comments/replies are counted. Composite identities are used
because the source tables lack unique platform comment IDs. No data is changed.
Outputs: descriptive_stats_verified.csv/.md and an audit JSON with input hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
L1_KEY = ["news_id", "comment_author", "comment_content", "comment_time"]
L2_KEY = L1_KEY + ["reply_author", "reply_content", "reply_time"]
COMMENT_COLUMNS = list(dict.fromkeys(L2_KEY + ["post_url", "post_time"]))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compute_class(news_file, comments_file):
    news = pd.read_excel(news_file, dtype=str, keep_default_na=False,
                         usecols=["news_id", "post_url", "post_time"])
    data = pd.read_excel(comments_file, dtype=str, keep_default_na=False,
                         usecols=COMMENT_COLUMNS)
    require(news.news_id.is_unique and news.post_url.is_unique,
            "News IDs and URLs must be unique")
    require(set(data.post_url) == set(news.post_url),
            "News and comment tables do not cover the same source posts")
    # Comment-table news_id is a local ID, unlike news-table news_id.
    # Join the original post URL, retaining each class's local namespace.
    links = data[["news_id", "post_url"]].drop_duplicates()
    require(links.news_id.is_unique and links.post_url.is_unique,
            "Comment news IDs must map one-to-one to news URLs")
    source_times = pd.to_datetime(news.set_index("post_url").post_time, errors="raise")
    times = pd.Series(links.post_url.map(source_times).to_numpy(), index=links.news_id)
    require(pd.to_datetime(data.post_time, errors="raise").eq(data.news_id.map(times)).all(),
            "Publication times differ between news and comment tables")

    l1 = data[data.comment_content.str.strip().ne("")].drop_duplicates(L1_KEY).copy()
    l2 = data[data.reply_content.str.strip().ne("")].drop_duplicates(L2_KEY).copy()
    l1_keys = set(map(tuple, l1[L1_KEY].to_numpy()))
    parents = set(map(tuple, l2[L1_KEY].to_numpy()))
    require(parents.issubset(l1_keys), "A reply lacks a matching nonempty L1 parent")

    l1_time = pd.to_datetime(l1.comment_time, errors="coerce")
    l2_time = pd.to_datetime(l2.reply_time, errors="coerce")
    parent_time = pd.to_datetime(l2.comment_time, errors="coerce")
    a = (l1_time - l1.news_id.map(times)).dt.total_seconds() / 3600
    b = (l2_time - parent_time).dt.total_seconds() / 3600
    quality = {"l1_missing_delay": int(a.isna().sum()),
               "l2_missing_delay": int(b.isna().sum()),
               "l1_negative_delay": int(a.lt(0).sum()),
               "l2_negative_delay": int(b.lt(0).sum())}
    require(not any(quality.values()), f"Invalid time observations: {quality}")

    events = pd.concat([
        pd.DataFrame({"news_id": l1.news_id, "time": l1_time}),
        pd.DataFrame({"news_id": l2.news_id, "time": l2_time}),
    ], ignore_index=True)
    last = events.groupby("news_id").time.max()
    life = (last - last.index.to_series().map(times)).dt.total_seconds() / 3600
    require(life.notna().all() and life.ge(0).all(), "Invalid lifespan")
    depth = pd.Series(0, index=links.news_id)
    depth.loc[list(set(l1.news_id))] = 1
    depth.loc[list(set(l2.news_id))] = 2
    characters = set("".join(l1.comment_content.tolist() + l2.reply_content.tolist()))
    sums = {
        "news": len(news), "l1": len(l1), "l2": len(l2),
        "no_comments": len(news) - l1.news_id.nunique(),
        "l1_without_l2": len(l1_keys - parents),
        "depth_sum": int(depth.sum()),
        "l1_length_sum": int(l1.comment_content.str.len().sum()),
        "l2_length_sum": int(l2.reply_content.str.len().sum()),
        "l1_delay_sum": float(a.sum()), "l1_delay_n": len(a),
        "l2_delay_sum": float(b.sum()), "l2_delay_n": len(b),
        "lifespan_sum": float(life.sum()), "lifespan_n": len(life),
    }
    return sums, characters, quality


def metrics(s, characters):
    return {
        "News posts": s["news"],
        "L1 comments": s["l1"],
        "L2 replies": s["l2"],
        "Posts without comments": s["no_comments"],
        "L1 comments without L2 replies": s["l1_without_l2"],
        "Avg. L1 delay (h)": s["l1_delay_sum"] / s["l1_delay_n"],
        "Avg. L2 delay (h)": s["l2_delay_sum"] / s["l2_delay_n"],
        "Avg. lifespan (h; posts with comments)": s["lifespan_sum"] / s["lifespan_n"],
        "Avg. width (L1 per post)": s["l1"] / s["news"],
        "Avg. tree depth": s["depth_sum"] / s["news"],
        "Avg. L1 comment length (chars)": s["l1_length_sum"] / s["l1"],
        "Avg. L2 reply length (chars)": s["l2_length_sum"] / s["l2"],
        "Character vocabulary size": len(characters),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=HERE.parent / "model" / "data" / "raw_data")
    parser.add_argument("--processed", action="store_true",
                        help="Read processed_<label>_*.xlsx from --data-dir")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    args = parser.parse_args()
    prefix = "processed_" if args.processed else ""
    totals, union, results = {}, set(), {}
    audit = {"input_sha256": {}, "class_accumulators": {}, "quality": {},
             "definitions": {
                 "l1_key": L1_KEY, "l2_key": L2_KEY,
                 "text": "Nonempty raw text; lengths include punctuation, spaces and URLs",
                 "lifespan": "Publication to last nonempty comment/reply; only posts with comments",
                 "width_depth": "All news; root depth 0 and observed maximum depth 2",
                 "total": "Pooled numerators/denominators; character set union",
                 "identity_limit": "Composite-key counts, not verified platform comment-ID counts",
             }}
    for label in ("fake", "real"):
        files = [args.data_dir / f"{prefix}{label}_{name}.xlsx" for name in ("news", "comments")]
        print(f"Reading {label} news and comments ...", flush=True)
        before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        sums, characters, quality = compute_class(*files)
        after = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        require(before == after, "An input file changed during computation")
        audit["input_sha256"].update(after)
        audit["class_accumulators"][label] = sums
        audit["quality"][label] = quality
        results[label.title()] = metrics(sums, characters)
        for key, value in sums.items():
            totals[key] = totals.get(key, 0) + value
        union |= characters
    results["Total"] = metrics(totals, union)
    audit["total_accumulators"] = totals
    audit["results"] = results
    table = pd.DataFrame(results)
    table.index.name = "Metric"
    lines = ["| Metric | Fake | Real | Total |", "|---|---:|---:|---:|"]
    for metric, row in table.iterrows():
        values = [f"{v:.2f}" if metric.startswith("Avg.") else f"{int(v):,}" for v in row]
        lines.append("| " + " | ".join([metric] + values) + " |")
    report = "\n".join(lines) + "\n"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "descriptive_stats_verified.csv", encoding="utf-8-sig", float_format="%.10f")
    (args.output_dir / "descriptive_stats_verified.md").write_text(report, encoding="utf-8")
    (args.output_dir / "descriptive_stats_verified_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
