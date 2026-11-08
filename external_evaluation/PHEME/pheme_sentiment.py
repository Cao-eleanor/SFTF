
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
SAVE_INTERVAL = 1000
LEVEL_CONFIG = {
    "0": ("pheme_news.csv", "news_id", "news", None),
    "1": ("pheme_nodes.csv", "node_id", "L1", "1"),
    "2": ("pheme_nodes.csv", "node_id", "L2", "2"),
}
RESULT_COLUMNS = {
    "negative_prob": "negative_pro",
    "neutral_prob": "neutral_pro",
    "positive_prob": "positive_pro",
    "sentiment": "sentiment_trend",
    "confidence": "confidence",
}
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
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def prepare_rows(level):
    filename, key, name, depth = LEVEL_CONFIG[level]
    rows = read_csv(RAW / filename)
    if depth is not None:
        rows = rows.loc[rows["depth"].eq(depth)].copy()
    rows = rows.reset_index(drop=True)
    if rows[key].eq("").any() or rows[key].duplicated().any():
        raise ValueError(f"Missing or duplicate {key} in {filename}")
    output = OUT / f"pheme_{name}_sentiment.csv"
    if output.exists():
        previous = read_csv(output)
        if key not in previous or previous[key].duplicated().any():
            raise ValueError(f"Invalid resume keys in {output}")
        if not set(previous[key]).issubset(set(rows[key])):
            raise ValueError(f"Resume file contains IDs outside the input: {output}")
        saved = previous.set_index(key)
        common = rows[key].isin(saved.index)
        # Preserve all existing fields, including text, original probabilities
        # and provenance. Reject changed input text for an existing annotation.
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
    if "clean_text" not in rows:
        rows["clean_text"] = rows["text"].map(clean_text)
    pending = rows.index[~rows["sentiment_trend"].isin(VALID_LABELS)].tolist()
    return rows, pending, output


def save_csv(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output, index=False, encoding="utf-8-sig")


def main():
    global API_KEY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=["0", "1", "2"], required=True,
                        help="0=news, 1=L1 comments, 2=L2 replies")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate inputs/resume state without API calls or writes")
    args = parser.parse_args()
    rows, pending, output = prepare_rows(args.level)
    print(f"Level {args.level}: {len(rows)} rows, {len(pending)} pending; output: {output}")
    if args.check_only or not pending:
        return
    API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    if not API_KEY:
        raise RuntimeError("Set DEEPSEEK_API_KEY before annotation")
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(get_sentiment_worker, i, rows.at[i, "clean_text"])
                   for i in pending]
        for future in as_completed(futures):
            idx, result = future.result()
            for original, column in RESULT_COLUMNS.items():
                value = result.get(original)
                rows.at[idx, column] = "" if value is None else str(value)
            if "sentiment_status" in rows:
                rows.at[idx, "sentiment_status"] = (
                    "complete" if rows.at[idx, "sentiment_trend"] in VALID_LABELS
                    else "missing_or_error")
            completed += 1
            print(f"Completed {completed}/{len(pending)}")
            if completed % SAVE_INTERVAL == 0:
                step = output.with_name(f"{output.stem}_step_{completed}.csv")
                save_csv(rows, step)
                save_csv(rows, output)
    save_csv(rows, output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
