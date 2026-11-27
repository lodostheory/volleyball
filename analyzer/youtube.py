"""유튜브 영상 받기 (yt-dlp)

- 분석에는 소리가 필요 없으므로 영상 트랙만 받는다 → ffmpeg 없이도 동작
- 1080p 이하에서 가장 좋은 화질, OpenCV 가 잘 읽는 H.264(avc1) 를 우선
- 본인 또는 권한이 있는 영상만 받으세요 (유튜브 이용약관)
"""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

YT_RE = re.compile(r'(?:youtu\.be/|youtube\.com/(?:watch\?(?:.*&)?v=|embed/|shorts/|live/))([\w-]{11})')


def youtube_id(url: str) -> str | None:
    m = YT_RE.search(url or '')
    if m:
        return m.group(1)
    return url if re.fullmatch(r'[\w-]{11}', url or '') else None


def _js_runtimes() -> dict:
    """유튜브 서명 해석에 쓸 자바스크립트 런타임 (deno 가 없으면 node)"""
    rt = {}
    for name in ('deno', 'node', 'bun'):
        exe = shutil.which(name)
        if exe:
            rt[name] = {'path': exe}
    return rt


def download(url: str, out_dir: Path, progress=None, max_height: int = 1080) -> tuple[Path, dict]:
    import yt_dlp
    vid = youtube_id(url)
    if not vid:
        raise ValueError('유튜브 주소가 아닙니다')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def hook(d):
        if progress and d.get('status') == 'downloading':
            tot = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
            progress(min(1.0, d.get('downloaded_bytes', 0) / tot) if tot else 0.0)

    h = int(max_height)
    opts = {
        'format': f'bv*[height<={h}][vcodec^=avc1]/bv*[height<={h}][vcodec^=vp9]/bv*[height<={h}]/b[height<={h}]/b',
        'outtmpl': str(out_dir / '%(id)s.%(ext)s'),
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'noprogress': True,
        'progress_hooks': [hook],
    }
    rts = _js_runtimes()
    if rts:
        opts['js_runtimes'] = rts
    opts.update(retries=10, fragment_retries=10, continuedl=True)
    last = None
    for attempt in range(5):        # 연결이 끊기거나 TLS 오류가 나도 받던 곳부터 이어서 다시 시도
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(f'https://www.youtube.com/watch?v={vid}', download=True)
                path = Path(ydl.prepare_filename(info))
            break
        except yt_dlp.utils.DownloadError as e:
            last = e
            msg = str(e)
            if not any(s in msg for s in ('SSL', 'timed out', 'Connection', 'Read timed', 'IncompleteRead', '10054', '10060')):
                raise
            time.sleep(2 + 3 * attempt)
    else:
        raise last
    meta = {k: info.get(k) for k in ('id', 'title', 'duration', 'fps', 'width', 'height', 'uploader', 'license', 'vcodec')}
    return path, meta
