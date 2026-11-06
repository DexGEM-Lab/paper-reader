from pathlib import Path
from unittest.mock import patch

from src.paper_reader.markdown_render import render_markdown
from types import SimpleNamespace
from src.paper_reader.web_resources import (
    WebResourceStore, _remove_widget_lists, _sanitize_extracted_markdown,
    _rewrite_inline_media, extract_rich_references,
)


def test_linked_images_and_video_schemes():
    image = render_markdown('[![a](https://img.test/a.png)](https://example.com)')
    assert '<a href="https://example.com"' in image and '<img src="https://img.test/a.png"' in image
    local = render_markdown('[![a](/files/a.png)](/files/a.mp4)')
    assert '<video controls preload="metadata" src="/files/a.mp4"></video>' in local
    remote = render_markdown('[watch](https://example.com/a.webm?q=1)')
    assert 'referrerpolicy="no-referrer"' in remote and '<video' in remote
    assert '<video' not in render_markdown('[x](javascript:alert.mp4)')
    assert '<img' not in render_markdown('![x](javascript:alert)')


def test_raw_video_whitelist():
    rendered = render_markdown('<video controls onclick="evil()" src="https://example.com/a.mp4"><source src="/files/a.webm" onerror="bad()"></video><script>x</script>')
    assert '<video' in rendered and '<source' in rendered
    assert 'onclick' not in rendered and 'onerror' not in rendered
    assert '&lt;script&gt;' in rendered
    assert '<video' not in render_markdown('<video src="javascript:alert(1)">')


def test_widget_filter_and_ordinary_list_survives():
    page = '<article><ul><li>Action</li><li>hold</li><li>Object</li><li>frying pan handle</li><li>Contact</li><li>yes</li><li>Visibility</li><li>visible</li></ul><ul><li>Color</li><li>red</li><li>Size</li><li>large</li></ul></article>'
    result = _remove_widget_lists(page)
    assert 'frying pan handle' not in result
    assert 'Color' in result and 'large' in result
    assert 'frying pan handle' not in _sanitize_extracted_markdown('- Action\n- hold\n- Object\n- frying pan handle\n- Contact\n- yes')
    assert 'Color' in _sanitize_extracted_markdown('- Color\n- red\n- Size\n- large')


def test_video_download_and_rewrite(tmp_path):
    html = '<video poster="/p.png"><source src="/demo.mp4"></video><a href="/other.webm">watch</a>'
    refs = extract_rich_references(html, base_url='https://example.com/a')
    assert len(refs['videos']) == 2
    store = WebResourceStore(tmp_path, video_max_bytes=123)
    response = SimpleNamespace(content_type='video/mp4', body=b'video')
    with patch('src.paper_reader.web_resources.fetch_public_url', return_value=response) as fetch:
        store._download_videos(refs['videos'], snapshot_path=tmp_path / 'x.md')
    assert fetch.call_args.kwargs['max_bytes'] == 123
    assert all(v['local_url'].startswith('/files/') for v in refs['videos'])
    result = _rewrite_inline_media('![HTML5 video](https://example.com/p.png)\n\n[watch](https://example.com/other.webm)', assets=[], videos=refs['videos'])
    assert result.count('/files/') == 2
    assert all(Path(tmp_path / v['local_url'].removeprefix('/files/')).read_bytes() == b'video' for v in refs['videos'])


def test_video_download_failure_keeps_remote(tmp_path):
    videos = [{'url': 'https://example.com/a.mp4', 'title': 'clip'}]
    store = WebResourceStore(tmp_path)
    with patch('src.paper_reader.web_resources.fetch_public_url', side_effect=ValueError('too large')):
        store._download_videos(videos, snapshot_path=tmp_path / 'x.md')
    assert videos[0]['download_error'] == 'too large'
    assert 'https://example.com/a.mp4' in _rewrite_inline_media('![clip](https://example.com/a.mp4)', assets=[], videos=videos)


def test_nested_video_poster_does_not_become_double_nested():
    videos = [{'url': 'https://example.com/a.mp4', 'local_url': '/files/a.mp4', 'title': 'clip'}]
    assets = [{'source_url': 'https://example.com/p.png', 'local_url': '/files/p.png', 'role': 'inline'}]
    result = _rewrite_inline_media('[![clip](https://example.com/p.png)](https://example.com/a.mp4)', assets=assets, videos=videos)
    assert result == '[![clip](/files/p.png)](/files/a.mp4)'
    assert '<video' in render_markdown(result)


def test_video_missing_from_extracted_body_is_preserved():
    videos = [{'url': 'https://example.com/a.mp4', 'local_url': '/files/a.mp4', 'title': 'clip', 'kind': 'video'}]
    content = _rewrite_inline_media('Article body', assets=[], videos=videos)
    assert '[▶ clip](/files/a.mp4)' in content
    assert '<video' in render_markdown(content)
