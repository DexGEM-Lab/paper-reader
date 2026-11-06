#!/usr/bin/env python3
"""Forward durable feedback changes to a running Pi session."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen


def poll(url, token, cursor):
    request = Request(f'{url.rstrip("/")}/internal/feedback/pending?cursor={cursor}',
                      headers={'Authorization': f'Bearer {token}', 'User-Agent': 'curl/8.0.1'})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='https://reader.dex-gem.ai')
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--cursor-file', type=Path, required=True)
    parser.add_argument('--to', required=True, help='running pi-notify session ID or name')
    parser.add_argument('--interval', type=int, default=180)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    while True:
        try:
            token = args.token_file.read_text().strip()
            if not token:
                raise ValueError('empty token')
            cursor = int(args.cursor_file.read_text().strip()) if args.cursor_file.exists() else 0
            result = poll(args.url, token, cursor)
            entries = result['entries']
            if entries:
                lines = [f"{e['issue']} [{e['status']}] {e['summary']}" for e in entries]
                subprocess.run(['pi-notify', 'send', '--to', args.to, '飞书反馈：\n' + '\n'.join(lines)], check=True)
                args.cursor_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = args.cursor_file.with_suffix('.tmp')
                tmp.write_text(str(result['cursor']))
                os.replace(tmp, args.cursor_file)
        except Exception as exc:
            print(f'feedback watcher: {exc}', file=sys.stderr, flush=True)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
