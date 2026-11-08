"""
Local table preparation utilities; no web requests or login required.

Run --help or <command> --help. Explicit input/output paths replace historical
desktop paths. Each command preserves source files and creates output parents.
"""

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

from common import OUTPUT, read_table, require_columns, write_table


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['links', 'sample', 'filter-comments', 'text-ids', 'prefix-urls', 'number-urls', 'convert', 'rename-comments', 'merge-years']:
        cmd = sub.add_parser(name)
        if name == 'links':
            cmd.add_argument('--input', type=Path, default=OUTPUT / 'resolved_reports.csv',
                             help='Source link table (default: outputs/resolved_reports.csv)')
            cmd.add_argument('--output', type=Path, default=OUTPUT / 'post_links.xlsx',
                             help='Output link table (default: outputs/post_links.xlsx)')
        else:
            cmd.add_argument('--input', type=Path, required=True, help='Source table; source directory for merge-years')
            cmd.add_argument('--output', type=Path, required=True, help='Output table; directory for merge-years/rename-comments')
        if name in ('links', 'prefix-urls', 'number-urls', 'rename-comments'):
            cmd.add_argument('--url-column', default={'links': 'post_url', 'number-urls': 'post_url'}.get(name, 'URL'))
        if name == 'sample':
            cmd.add_argument('--count', type=int, default=764)
            cmd.add_argument('--seed', type=int, default=42)
        if name == 'filter-comments':
            cmd.add_argument('--news', type=Path, required=True)
            cmd.add_argument('--news-column', default='post_url')
            cmd.add_argument('--comment-column', default='post_url')
        if name in ('text-ids', 'prefix-urls'):
            cmd.add_argument('--id-column', default='id')
        if name == 'number-urls':
            cmd.add_argument('--id-column', default='ids')
        if name == 'rename-comments':
            cmd.add_argument('--source-dir', type=Path, required=True)
        if name == 'merge-years':
            cmd.add_argument('--year-pattern', default=r'(\d{4})年', help='Regex with year as capture group 1')
    return parser


def prepare(args):
    """Execute one local operation and return the number of output rows/files."""
    if args.input.resolve() == args.output.resolve():
        raise ValueError('Input and output must differ')
    if args.command == 'merge-years':
        groups = {}
        for path in sorted(args.input.glob('*.xlsx')):
            if path.name.startswith('~$'):
                continue
            match = re.search(args.year_pattern, path.name)
            if match:
                groups.setdefault(match.group(1), []).append(path)
        for year, paths in groups.items():
            if not year.isdigit():
                raise ValueError('Year capture must contain digits only')
            write_table(pd.concat([read_table(p) for p in paths], ignore_index=True), args.output / f'{year}_combined.xlsx')
        return len(groups)
    frame = read_table(args.input)
    if args.command == 'links':
        from common import parse_link
        require_columns(frame, [args.url_column])
        links = []
        for value in frame[args.url_column]:
            text = str(value).strip()
            if not text:
                continue
            try:
                links.append(parse_link(text)[2])
            except ValueError as exc:
                print(f'Warning: skipped invalid URL {text!r}: {exc}')
        links = list(dict.fromkeys(links))
        frame = pd.DataFrame({'id': range(1, len(links) + 1), 'post_url': links})
    elif args.command == 'sample':
        if args.count < 0:
            raise ValueError('Count must not be negative')
        frame = frame.sample(n=args.count, random_state=args.seed) if len(frame) >= args.count else frame
    elif args.command == 'filter-comments':
        news = read_table(args.news)
        require_columns(frame, [args.comment_column])
        require_columns(news, [args.news_column])
        frame = frame[frame[args.comment_column].isin(set(news[args.news_column]) - {''})]
    elif args.command == 'text-ids':
        require_columns(frame, [args.id_column])
        frame[args.id_column] = frame[args.id_column].astype(str)
    elif args.command == 'prefix-urls':
        require_columns(frame, [args.id_column, args.url_column])
        frame[args.url_column] = frame[args.id_column] + '_' + frame[args.url_column]
    elif args.command == 'number-urls':
        require_columns(frame, [args.url_column])
        frame[args.id_column] = pd.factorize(frame[args.url_column].replace('', pd.NA))[0] + 1
    elif args.command == 'rename-comments':
        require_columns(frame, [args.url_column])
        mapping = {}
        for value in frame[args.url_column]:
            if not re.fullmatch(r'[A-Za-z0-9_]+', value):
                raise ValueError('Mapping values must be safe filename stems, e.g. 1_Q7gxkFAEm')
            suffix = value.split('_')[-1]
            if suffix in mapping and mapping[suffix] != value:
                raise ValueError('Conflicting URL suffix mapping')
            mapping[suffix] = value
        args.output.mkdir(parents=True, exist_ok=True)
        count = 0
        for path in sorted(args.source_dir.glob('*.xlsx')):
            match = re.fullmatch(r'([A-Za-z0-9]+)_(\d+)\.xlsx', path.name)
            if match and match[1] in mapping:
                target = args.output / f'{mapping[match[1]]}_{match[2]}.xlsx'
                if not target.exists():
                    shutil.copy2(path, target)
                    count += 1
        return count
    write_table(frame, args.output)
    return len(frame)


def main():
    args = build_parser().parse_args()
    count = prepare(args)
    print(f'Completed {args.command}: {count} rows/files written to {args.output}')


if __name__ == '__main__':
    main()
