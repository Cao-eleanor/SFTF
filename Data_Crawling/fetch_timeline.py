"""
Collect one account's posts by month; export monthly and combined CSV files.

Account UID, year, output directory, and private cookie source are CLI options.
Month boundaries use UTC+08:00. Responses are filtered again by posting time.
Only post text and interaction counts are collected, not comment bodies.
"""

import argparse
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from common import OUTPUT, get_json, load_cookie, make_session, manifest, write_table

TARGET_UID = ""
LOCAL_TZ = timezone(timedelta(hours=8))
COLUMNS = ['id', 'post_url', 'post_time', 'repost_count', 'comment_count', 'like_count', 'post_content']


def month_bounds(year, month):
    """Return inclusive-start/exclusive-end epoch seconds for a local month."""
    start = datetime(year, month, 1, tzinfo=LOCAL_TZ)
    end = datetime(year + (month == 12), 1 if month == 12 else month + 1, 1, tzinfo=LOCAL_TZ)
    return int(start.timestamp()), int(end.timestamp())


def in_month(record, start, end):
    try:
        stamp = datetime.strptime(record['created_at'], '%a %b %d %H:%M:%S %z %Y')
    except (KeyError, TypeError, ValueError):
        try:
            stamp = datetime.strptime(record['created_at'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=LOCAL_TZ)
        except (KeyError, TypeError, ValueError):
            return False
    return start <= stamp.timestamp() < end


def collect_month(session, uid, year, month, max_pages, delay):
    start, end = month_bounds(year, month)
    records = {}
    stop_reason = 'page_limit'
    for page in range(1, max_pages + 1):
        payload = get_json(session, 'https://weibo.com/ajax/statuses/searchProfile', {
            'uid': uid, 'page': page, 'starttime': start, 'endtime': end - 1,
            'hasori': 1, 'hasret': 1, 'hastext': 1, 'haspic': 1, 'hasvideo': 1, 'hasmusic': 1,
        })
        container = payload.get('data')
        if not isinstance(container, dict) or not isinstance(container.get('list'), list):
            raise ValueError('Timeline response has no data.list; check login or API changes')
        posts = container['list']
        if not posts:
            stop_reason = 'empty_page'
            break
        for post in posts:
            if in_month(post, start, end):
                parsed = parse_weibo(post)
                if parsed:
                    records[parsed['id']] = parsed
        print(f'{year}-{month:02d}, page {page}: {len(records)} unique posts')
        time.sleep(delay)
    return pd.DataFrame(records.values(), columns=COLUMNS), stop_reason


def main():
    global TARGET_UID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uid', required=True, help='Numeric account ID')
    parser.add_argument('--year', required=True, type=int)
    parser.add_argument('--months', nargs='+', type=int, default=list(range(1, 13)))
    parser.add_argument('--output-dir', type=Path, default=OUTPUT / 'timeline')
    parser.add_argument('--cookie-file', type=Path, help='Private file; otherwise WEIBO_COOKIE')
    parser.add_argument('--max-pages', type=int, default=200)
    parser.add_argument('--delay', type=float, default=8)
    parser.add_argument('--month-delay', type=float, default=60)
    args = parser.parse_args()
    if not args.uid.isdigit() or not 1970 <= args.year <= 9998:
        parser.error('Require a numeric UID and a year between 1970 and 9998')
    if any(m not in range(1, 13) for m in args.months) or args.max_pages < 1 or min(args.delay, args.month_delay) < 0:
        parser.error('Invalid months, page limit, or delays')
    TARGET_UID = args.uid
    session = make_session(load_cookie(args.cookie_file))
    frames, files, stops = [], [], {}
    months = sorted(set(args.months))
    settings = {'uid': args.uid, 'year': args.year, 'requested_months': months,
                'max_pages': args.max_pages, 'delay': args.delay,
                'month_delay': args.month_delay, 'month_stop_reasons': stops, 'status': 'running'}
    manifest(args.output_dir, Path(__file__).name, settings, files)
    try:
        for index, month in enumerate(months):
            frame, reason = collect_month(session, args.uid, args.year, month, args.max_pages, args.delay)
            frame = frame.sort_values('post_time', ascending=False)
            path = args.output_dir / f'{args.uid}_{args.year}_{month:02d}.csv'
            write_table(frame, path)
            frames.append(frame)
            files.append(path)
            stops[str(month)] = reason
            manifest(args.output_dir, Path(__file__).name, settings, files)
            if index + 1 < len(months):
                time.sleep(args.month_delay)
        merged = pd.concat(frames, ignore_index=True).drop_duplicates('id').sort_values('post_time', ascending=False)
        combined = args.output_dir / f'{args.uid}_{args.year}_combined.csv'
        write_table(merged, combined)
        files.append(combined)
        settings['status'] = 'complete'
    except BaseException:
        settings['status'] = 'incomplete'
        raise
    finally:
        session.close()
        manifest(args.output_dir, Path(__file__).name, settings, files)


def parse_weibo(weibo_json):
    """Extract post metadata, interaction counts, and text from an API record."""
    try:
        weibo_id = str(weibo_json.get('id', '') or weibo_json.get('mid', '') or weibo_json.get('idstr', ''))
        
        if not weibo_id:
            return None
        
        user_info = weibo_json.get('user', {}) or {}
        user_id = user_info.get('id', TARGET_UID)
        mblogid = weibo_json.get('mblogid', '')
        
        if mblogid:
            original_url = f"https://weibo.com/{user_id}/{mblogid}"
        else:
            original_url = f"https://weibo.com/{user_id}/{weibo_id}"
        
        created_at = weibo_json.get('created_at', '')
        post_time_str = parse_weibo_time(created_at)
        
        content = weibo_json.get('text_raw', '') or ''
        if not content:
            content = weibo_json.get('text', '') or ''
            content = re.sub(r'<[^>]+>', '', content)
        content = content.strip()
        
        retweeted = weibo_json.get('retweeted_status')
        if retweeted:
            rt_user = retweeted.get('user', {}) or {}
            rt_name = rt_user.get('screen_name', 'Unknown user')
            rt_content = retweeted.get('text_raw', '') or retweeted.get('text', '')
            rt_content = re.sub(r'<[^>]+>', '', rt_content).strip()
            if rt_content:
                content = f"{content} //@{rt_name}: {rt_content}"
        
        reposts_count = weibo_json.get('reposts_count', 0) or 0
        comments_count = weibo_json.get('comments_count', 0) or 0
        attitudes_count = weibo_json.get('attitudes_count', 0) or 0
        
        return {
            'id': weibo_id,
            'post_url': original_url,
            'post_time': post_time_str,
            'repost_count': int(reposts_count),
            'comment_count': int(comments_count),
            'like_count': int(attitudes_count),
            'post_content': content
        }
        
    except Exception as e:
        return None

def parse_weibo_time(time_str):
    """Format recognized timestamps; retain unrecognized strings unchanged."""
    try:
        if not time_str:
            return "Unknown time"
        
        try:
            dt = datetime.strptime(time_str, '%a %b %d %H:%M:%S %z %Y')
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
        
        try:
            dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            return time_str
        except:
            pass
        
        return time_str
    except:
        return time_str


if __name__ == "__main__":
    main()
