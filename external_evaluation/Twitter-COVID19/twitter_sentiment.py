# -*- coding: utf-8 -*-
"""
为 Twitter-COVID19 提取 DeepSeek 情感，补齐 04_sentiment_features 覆盖不到的节点。

仿照 PHEME 的 pheme_sentiment.py。三档：
  --level 0  新闻（key=news_id，来自 twitter_news.csv）
  --level 1  一级评论（key=node_id，来自 twitter_nodes.csv 的 depth==1）
  --level 2  二级评论（key=node_id，来自 twitter_nodes.csv 的 depth==2）

本脚本会自动 resume（保留已有情感，只补标 sentiment_trend 缺失的节点）。

时间用 raw_data 里的 delay_seconds（源数据无绝对时间戳）；情感 schema 与 PHEME 一致：
positive_pro / neutral_pro / negative_pro / sentiment_trend / confidence。

两种运行模式：

  1) 全量模式（默认，无分片参数）——一次性标注某层全部待标节点，写主文件：
       export DEEPSEEK_API_KEY=...
       python twitter_sentiment.py --level 0
       python twitter_sentiment.py --level 1
       python twitter_sentiment.py --level 2

  2) 分片模式（加 --start/--end/--part）——只处理 pending 列表的 [start, end) 一段，
     写入独立文件 sentiment_data/twitter_{name}_sentiment_{part}.csv，互不冲突，
     可同时开多个进程并行跑。全部跑完后将各分片文件合并回主文件：
       python twitter_sentiment.py --level 1 --start 0       --end 50000  --part L1_1
       python twitter_sentiment.py --level 1 --start 50000   --end 100000 --part L1_2
       python twitter_sentiment.py --level 1 --start 100000  --end 150000 --part L1_3
       python twitter_sentiment.py --level 1 --start 150000  --end 200000 --part L1_4

  --check-only 只校验输入/resume 状态并打印统计，不调用 API、不写文件。
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw_data"
OUT = HERE / "sentiment_data"
API_URL = "https://api.deepseek.com/chat/completions"
API_KEY = ""
MAX_WORKERS = 3
SAVE_INTERVAL = 10000                 # 每完成 1 万条写一次盘
MAX_CONSECUTIVE_FAILURES = 100        # 连续失败 100 条则中止（防 key 失效/限流空转）
LEVEL_CONFIG = {
    "0": ("twitter_news.csv", "news_id", "news", None),
    "1": ("twitter_nodes.csv", "node_id", "L1", "1"),
    "2": ("twitter_nodes.csv", "node_id", "L2", "2"),
}
RESULT_COLUMNS = {
    "negative_prob": "negative_pro",
    "neutral_prob": "neutral_pro",
    "positive_prob": "positive_pro",
    "sentiment": "sentiment_trend",
    "confidence": "confidence",
}
# 分片输出/合并用到的情感列（与主文件列名一致）
SENTIMENT_COLUMNS = ["positive_pro", "neutral_pro", "negative_pro", "sentiment_trend", "confidence"]
VALID_LABELS = {"positive", "neutral", "negative"}


def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text)
    text = re.sub('http\\S+', '', text)
    text = re.sub('@\\w+', '@USER', text)
    text = re.sub('\\s+', ' ', text)
    return text.strip()


def get_sentiment_worker(idx, text):
    if pd.isna(text):
        text = ''
    text = str(text)
    if text.strip() == '':
        return (idx, {'positive_prob': 0, 'neutral_prob': 1, 'negative_prob': 0, 'sentiment': 'neutral', 'confidence': 1})
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {API_KEY}'}
    payload = {'model': 'deepseek-chat', 'messages': [{'role': 'system', 'content': '\n\n            你是社交媒体文本情感分析专家。\n\n            分析文本情感。\n\n            只返回JSON,不要输出解释。\n\n            格式:\n\n                {\n\n                    "positive_prob":0.0,\n\n                    "neutral_prob":0.0,\n\n                    "negative_prob":0.0,\n\n                    "sentiment":"positive/neutral/negative",\n\n                    "confidence":0.0\n\n                }\n\n            要求：\n\n                1. 三个概率范围0-1\n\n                2. 三个概率之和等于1\n\n                3. confidence范围0-1\n\n            '}, {'role': 'user', 'content': text}], 'temperature': 0.1, 'max_tokens': 200}
    for attempt in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=(10, 30))
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            content = content.replace('```json', '')
            content = content.replace('```', '')
            data = json.loads(content.strip())
            time.sleep(0.2)
            return (idx, {'positive_prob': data.get('positive_prob'), 'neutral_prob': data.get('neutral_prob'), 'negative_prob': data.get('negative_prob'), 'sentiment': data.get('sentiment'), 'confidence': data.get('confidence')})
        except Exception as e:
            print('\n')
            print('=' * 80)
            print('DeepSeek调用失败')
            print('节点:', idx)
            print('尝试:', attempt + 1)
            print('错误:', e)
            print('=' * 80)
            time.sleep(2)
    return (idx, {'positive_prob': None, 'neutral_prob': None, 'negative_prob': None, 'sentiment': 'error', 'confidence': None})


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def prepare_rows(level):
    """读取输入 + 从主文件 resume，返回 (rows, pending, name, key)。"""
    filename, key, name, depth = LEVEL_CONFIG[level]
    rows = read_csv(RAW / filename)
    if depth is not None:
        rows = rows.loc[rows["depth"].eq(depth)].copy()
    rows = rows.reset_index(drop=True)
    if rows[key].eq("").any() or rows[key].duplicated().any():
        raise ValueError(f"Missing or duplicate {key} in {filename}")
    base_output = OUT / f"twitter_{name}_sentiment.csv"
    if base_output.exists():
        previous = read_csv(base_output)
        if key not in previous or previous[key].duplicated().any():
            raise ValueError(f"Invalid resume keys in {base_output}")
        if not set(previous[key]).issubset(set(rows[key])):
            raise ValueError(f"Resume file contains IDs outside the input: {base_output}")
        saved = previous.set_index(key)
        common = rows[key].isin(saved.index)
        if "text" in saved:
            old_text = rows.loc[common, key].map(saved["text"])
            if not old_text.equals(rows.loc[common, "text"]):
                raise ValueError("Input text differs from existing sentiment data")
        for column in previous.columns:
            if column == key:
                continue
            if column not in rows:
                rows[column] = ""
            rows.loc[common, column] = rows.loc[common, key].map(saved[column])
    for column in RESULT_COLUMNS.values():
        if column not in rows:
            rows[column] = ""
    # clean_text 一律按 text 确定性重算：resume 时主文件里待标行的 clean_text 可能是空串，
    # 沿用会导致对空文本标注（全部判 neutral）。
    rows["clean_text"] = rows["text"].map(clean_text)
    pending = rows.index[~rows["sentiment_trend"].isin(VALID_LABELS)].tolist()
    return rows, pending, name, key


def save_full(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output, index=False, encoding="utf-8-sig")


def save_part(rows, positions, output, key):
    cols = [key] + SENTIMENT_COLUMNS
    cols = [c for c in cols if c in rows.columns]
    sub = rows.loc[positions, cols].copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output, index=False, encoding="utf-8-sig")


def run(level, start=0, end=None, part=None):
    global API_KEY
    rows, pending, name, key = prepare_rows(level)
    total_pending = len(pending)
    shard = part is not None

    if shard:
        slice_positions = pending[start:end]
        output = OUT / f"twitter_{name}_sentiment_{part}.csv"
        # resume：分片文件里已经标成 valid 的 key 不再重复标注；"error" 行会重试。
        done_keys = set()
        if output.exists():
            prev = read_csv(output)
            if key not in prev or "sentiment_trend" not in prev:
                raise ValueError(f"Invalid part file (missing columns): {output}")
            done_keys = set(prev.loc[prev["sentiment_trend"].isin(VALID_LABELS), key])
        todo = [i for i in slice_positions if rows.at[i, key] not in done_keys]
        print(f"Level {level} [{start}:{end}]  total_pending={total_pending}  "
              f"slice={len(slice_positions)}  todo={len(todo)}")
        print(f"output: {output}")
    else:
        slice_positions = pending
        output = OUT / f"twitter_{name}_sentiment.csv"
        todo = pending
        print(f"Level {level}: {len(rows)} rows, {len(pending)} pending; output: {output}")

    if not todo:
        print("没有待标任务，跳过。")
        return 0

    API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    if not API_KEY:
        raise RuntimeError("Set DEEPSEEK_API_KEY before annotation")

    completed = 0
    consecutive_failures = 0
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    try:
        futures = [executor.submit(get_sentiment_worker, i, rows.at[i, "clean_text"])
                   for i in todo]
        for future in as_completed(futures):
            idx, result = future.result()
            for original, column in RESULT_COLUMNS.items():
                value = result.get(original)
                rows.at[idx, column] = "" if value is None else str(value)
            if "sentiment_status" in rows:
                rows.at[idx, "sentiment_status"] = (
                    "complete" if rows.at[idx, "sentiment_trend"] in VALID_LABELS
                    else "missing_or_error")
            if rows.at[idx, "sentiment_trend"] == "error":
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            completed += 1
            label = part or "full"
            print(f"[{label}] Completed {completed}/{len(todo)} (连续失败 {consecutive_failures})")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"[ABORT] 连续 {consecutive_failures} 条情感提取失败，中止并保存进度。")
                if shard:
                    save_part(rows, slice_positions, output, key)
                else:
                    save_full(rows, output)
                break
            if completed % SAVE_INTERVAL == 0:
                if shard:
                    save_part(rows, slice_positions, output, key)
                else:
                    step = output.with_name(f"{output.stem}_step_{completed}.csv")
                    save_full(rows, step)
                    save_full(rows, output)
    finally:
        try:
            executor.shutdown(wait=False, cancel_futures=True)   # Python 3.9+
        except TypeError:
            executor.shutdown(wait=False)
    if shard:
        save_part(rows, slice_positions, output, key)
    else:
        save_full(rows, output)
    print(f"Saved: {output}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=["0", "1", "2"], required=True,
                        help="0=news, 1=L1 comments, 2=L2 replies")
    parser.add_argument("--start", type=int, default=0,
                        help="分片：pending 列表起始下标（含）；缺省=全量模式")
    parser.add_argument("--end", type=int, default=None,
                        help="分片：pending 列表结束下标（不含）")
    parser.add_argument("--part", default=None,
                        help="分片标签，输出文件为 twitter_{name}_sentiment_{part}.csv；缺省=全量模式")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate inputs/resume state without API calls or writes")
    args = parser.parse_args()
    if args.check_only:
        rows, pending, name, key = prepare_rows(args.level)
        print(f"Level {args.level}: {len(rows)} rows, {len(pending)} pending; "
              f"output: twitter_{name}_sentiment.csv")
        return
    return run(args.level, args.start, args.end, args.part)


if __name__ == "__main__":
    raise SystemExit(main())
