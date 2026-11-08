"""
Collect posts and two levels of comments from a CSV/XLSX link list.

Run --help for paths and authentication. Each run requires an empty output
directory to prevent mixing or duplicating old comments. IDs are stored as
strings. Original Chinese column names remain compatible with legacy tables.
"""

import argparse
import hashlib
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from common import OUTPUT, get_json, load_cookie, make_session, manifest, parse_link, read_table, require_columns, write_table

NEWS = ['post_url', 'id', 'post_author', 'post_time', 'comment_count', 'like_count', 'repost_count', 'post_content', 'image_url', 'video_url', 'repost_url']
COMMENTS = ['id', 'comment_author', 'comment_time', 'comment_likes', 'reply_count', 'comment_content', 'comment_author_url']
REPLIES = ['id', 'comment_author', 'comment_time', 'comment_content', 'comment_author_url']
COMBINED = ['post_url', 'post_author', 'post_time', 'post_content', 'image_url', 'video_url',
            'repost_url', 'repost_count', 'comment_count', 'like_count', 'comment_author', 'comment_author_url', 'comment_content', 'comment_time',
            'comment_likes', 'reply_author', 'reply_time', 'reply_content', 'reply_author_url']


def timestamp(value):
    """Format Weibo timestamps; retain unrecognized strings unchanged."""
    value = (value or '').strip()
    if not value:
        return 'Unknown time'
    for fmt in ('%a %b %d %H:%M:%S %z %Y', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(value, fmt).strftime('%Y-%m-%d %H:%M:%S')
        except (TypeError, ValueError):
            continue
    return value


def profile(user):
    link = str((user or {}).get('profile_url', '') or '')
    return link if link.startswith('http') else 'https://weibo.com' + link


def post_record(payload, url):
    user = payload['user']
    retweet = payload.get('retweeted_status') or {}
    repost_url = ''
    if retweet.get('mblogid') and retweet.get('user', {}).get('id'):
        repost_url = f"https://weibo.com/{retweet['user']['id']}/{retweet['mblogid']}"
    video = (payload.get('page_info') or {}).get('media_info') or {}
    return [url, str(payload['id']), user['screen_name'], timestamp(payload['created_at']),
            int(payload.get('comments_count', 0) or 0), int(payload.get('attitudes_count', 0) or 0), int(payload.get('reposts_count', 0) or 0),
            payload.get('text_raw', ''), ' '.join(f'https://wx1.sinaimg.cn/orj360/{pic}.jpg' for pic in payload.get('pic_ids', [])),
            video.get('h265_mp4_hd', ''), repost_url]


def comment_pages(session, identity, uid, replies, delay, max_pages):
    """Yield unique API comments, stopping on an exhausted or repeated cursor."""
    cursor = ''
    cursors, ids = set(), set()
    for _ in range(max_pages):
        params = {'id': identity, 'uid': uid, 'is_reload': 1, 'is_show_bulletin': 2,
                  'is_mix': int(replies), 'fetch_level': int(replies), 'count': 20, 'locale': 'zh-CN'}
        if cursor or replies:
            params['max_id'] = cursor or 0
        if cursor:
            params['flow'] = 0
        payload = get_json(session, 'https://weibo.com/ajax/statuses/buildComments', params)
        records = payload.get('data')
        if not isinstance(records, list):
            raise ValueError('Comments response has no data list')
        for record in records:
            if not isinstance(record, dict) or not record.get('id'):
                continue
            key = str(record['id'])
            if key not in ids:
                ids.add(key)
                yield record
        cursor = str(payload.get('max_id', 0))
        if not records or cursor in ('0', ''):
            return
        if cursor in cursors:
            raise RuntimeError('Repeated comment cursor; collection is incomplete')
        cursors.add(cursor)
        time.sleep(delay)
    raise RuntimeError('Comment page limit reached; use a higher --max-pages with a new output directory')


def combined_rows(news, comments, replies):
    """Join replies to parent IDs; retain posts and comments without replies."""
    base = [news[0], news[2], news[3], news[7], news[8], news[9], news[10], news[6], news[4], news[5]]
    if not comments:
        return [base + [None] * 9]
    grouped = {}
    for row in replies:
        grouped.setdefault(str(row[0]), []).append(row[1:])
    result = []
    for identity, author, when, likes, count, body, link in comments:
        first = [author, link, body, when, likes]
        for reply in grouped.get(str(identity), [[None] * 4]):
            result.append(base + first + reply)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=OUTPUT / 'post_links.xlsx',
                        help='Link list (default: outputs/post_links.xlsx)')
    parser.add_argument('--url-column', default='post_url')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT / 'threads')
    parser.add_argument('--cookie-file', type=Path)
    parser.add_argument('--delay', type=float, default=2)
    parser.add_argument('--max-pages', type=int, default=1000)
    args = parser.parse_args()
    if args.delay < 0 or args.max_pages < 1:
        parser.error('Invalid delay or page limit')
    frame = read_table(args.input)
    require_columns(frame, [args.url_column])
    links = list(dict.fromkeys(parse_link(v) for v in frame[args.url_column] if str(v).strip()))
    if not links:
        parser.error('No nonempty post URLs found')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new or empty output directory; previous results are not overwritten')
    session = make_session(load_cookie(args.cookie_file))
    comment_dir = args.output_dir / 'comment'
    comment_dir.mkdir(parents=True, exist_ok=True)
    news_rows, joined, files = [], [], []
    settings = {'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'url_column': args.url_column, 'delay': args.delay, 'max_pages': args.max_pages,
                'requested_posts': len(links), 'completed_posts': 0, 'status': 'running'}
    try:
        for uid, bid, url in links:
            payload = get_json(session, 'https://weibo.com/ajax/statuses/show', {'id': bid, 'locale': 'zh-CN', 'isGetLongText': 'true'})
            news = post_record(payload, url)
            comments, replies = [], []
            for first in comment_pages(session, news[1], uid, False, args.delay, args.max_pages) if news[4] else []:
                parent_id = str(first.get('id'))
                comments.append([parent_id, (first.get('user') or {}).get('screen_name', ''), timestamp(first.get('created_at', '')),
                                 first.get('like_counts', 0), first.get('total_number', 0), first.get('text_raw', ''), profile(first.get('user'))])
                if first.get('total_number', 0):
                    time.sleep(args.delay)
                    for reply in comment_pages(session, parent_id, uid, True, args.delay, args.max_pages):
                        replies.append([parent_id, (reply.get('user') or {}).get('screen_name', ''), timestamp(reply.get('created_at', '')),
                                        reply.get('text_raw', ''), profile(reply.get('user'))])
            for suffix, rows, columns in [('1', comments, COMMENTS), ('2', replies, REPLIES)]:
                path = comment_dir / f'{bid}_{suffix}.xlsx'
                write_table(pd.DataFrame(rows, columns=columns), path)
                files.append(path)
            news_rows.append(news)
            joined.extend(combined_rows(news, comments, replies))
            write_table(pd.DataFrame(news_rows, columns=NEWS), comment_dir / 'text_info.xlsx')
            write_table(pd.DataFrame(joined, columns=COMBINED), args.output_dir / 'data.xlsx')
            settings['completed_posts'] = len(news_rows)
            print(f'Completed {len(news_rows)}/{len(links)} posts')
            time.sleep(args.delay)
        settings['status'] = 'complete'
    except BaseException:
        settings['status'] = 'incomplete'
        raise
    finally:
        session.close()
        manifest(args.output_dir, Path(__file__).name, settings,
                 files + [comment_dir / 'text_info.xlsx', args.output_dir / 'data.xlsx'])


if __name__ == '__main__':
    main()
