"""배구 영상 인식 로컬 서버 — 사이트의 「📹 경기 분석」 탭이 이 서버를 불러 씁니다.

실행: run_server.bat 더블클릭 (처음 한 번은 라이브러리를 설치)  또는  python server.py [--port 8765]
이 PC 안(127.0.0.1)에서만 열리고, VOLLYBALL MANIA 사이트와 이 PC 의 주소에서 온 요청만 받습니다.

API
  GET  /health                     서버·장치 상태
  POST /videos {url}               유튜브 영상 받기 시작 → {id}
  GET  /videos                     받은 영상 목록 (+ videos 폴더에 넣은 영상)
  GET  /videos/<id>                받기 진행·영상 정보
  GET  /videos/<id>/frame?t=초&w=폭  그 시각 화면(JPEG) — 코트 모서리 찍기용
  POST /jobs {video, corners, court, fps, start, end, model, tiles}  분석 시작 → {id}
  GET  /jobs/<id>                  진행률 · 결과(이벤트)
  POST /jobs/<id>/cancel           분석 취소
  GET  /jobs/<id>/excel            엑셀 리포트 내려받기
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import volleyball_analyzer as va  # noqa: E402
from youtube import download, youtube_id  # noqa: E402

VERSION = '1.0'
DATA = HERE / 'data'
VID_DIR, RES_DIR, LOCAL_DIR = DATA / 'videos', DATA / 'results', HERE / 'videos'
ALLOWED_ORIGIN = re.compile(r'^(https://lodostheory\.github\.io|https?://(localhost|127\.0\.0\.1)(:\d+)?)$')
VIDEO_EXT = {'.mp4', '.mkv', '.mov', '.webm', '.avi', '.m4v'}
# 사이트 화면도 이 서버에서 /app/ 으로 제공한다. 공개 사이트(https)에서 이 PC 로의 접근을 브라우저가 막아도
# 화면과 서버가 같은 주소가 되어 그대로 동작한다. 저장소 맨 위 폴더의 파일만, 정해진 종류만 내준다
SITE = HERE.parent
SITE_TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.json': 'application/json; charset=utf-8',
              '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon'}

videos: dict[str, dict] = {}
jobs: dict[str, dict] = {}
DEVICE, GPU = 'cpu', ''                           # 서버를 켤 때 한 번 정한다
state_lock = threading.Lock()
run_lock = threading.Lock()                        # 분석은 한 번에 하나씩 (GPU·CPU 를 다 쓰므로)
PORT = 8765


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------------------------------------
# 영상
# ---------------------------------------------------------------------------------------------
def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {'fps': round(fps, 3), 'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'frames': n, 'duration': round(n / fps, 2) if fps else 0, 'readable': bool(cap.isOpened())}
    finally:
        cap.release()


def scan_videos():
    """이미 받아 둔 영상과 videos 폴더에 넣은 영상을 목록에 올린다"""
    for d, prefix in ((VID_DIR, ''), (LOCAL_DIR, 'local-')):
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.suffix.lower() not in VIDEO_EXT or f.name.endswith('.part'):
                continue
            vid = prefix + (f.stem if not prefix else re.sub(r'[^\w.-]', '_', f.name))
            if vid in videos:
                continue
            meta_f = f.with_suffix('.json')
            meta = json.loads(meta_f.read_text(encoding='utf-8')) if meta_f.exists() else {}
            videos[vid] = {'id': vid, 'status': 'ready', 'progress': 1.0, 'path': str(f), 'title': meta.get('title') or f.name,
                           'local': bool(prefix), **probe(f)}


def start_download(url: str) -> dict:
    vid = youtube_id(url)
    if not vid:
        raise ValueError('유튜브 주소가 아닙니다')
    with state_lock:
        v = videos.get(vid)
        if v and v['status'] in ('ready', 'downloading'):
            return v
        v = videos[vid] = {'id': vid, 'status': 'downloading', 'progress': 0.0, 'title': '', 'url': url}

    def run():
        try:
            log('영상 받기 시작', url)
            path, meta = download(url, VID_DIR, progress=lambda f: v.update(progress=round(f, 3)))
            path.with_suffix('.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
            info = probe(path)
            if not info['readable'] or not info['frames']:
                raise RuntimeError('받은 영상을 OpenCV 로 읽을 수 없습니다 (코덱 문제)')
            v.update(status='ready', progress=1.0, path=str(path), title=meta.get('title') or vid, **info)
            log('영상 받기 완료', path.name, f"{info['width']}x{info['height']} {info['fps']}fps {info['duration']}초")
        except Exception as e:  # noqa: BLE001
            v.update(status='error', error=str(e)[:500])
            log('영상 받기 실패', e)

    threading.Thread(target=run, daemon=True).start()
    return v


def frame_jpeg(v: dict, t: float, width: int) -> bytes:
    cap = cv2.VideoCapture(v['path'])
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
        ok, fr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, fr = cap.read()
        if not ok:
            raise RuntimeError('화면을 읽을 수 없습니다')
    finally:
        cap.release()
    if width and fr.shape[1] > width:
        fr = cv2.resize(fr, (width, round(fr.shape[0] * width / fr.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.imencode('.jpg', fr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


# ---------------------------------------------------------------------------------------------
# 분석 작업
# ---------------------------------------------------------------------------------------------
def start_job(body: dict) -> dict:
    v = videos.get(str(body.get('video', '')))
    if not v or v['status'] != 'ready':
        raise ValueError('준비된 영상이 아닙니다')
    corners = body.get('corners')
    if not (isinstance(corners, list) and len(corners) == 4 and all(isinstance(p, list) and len(p) == 2 for p in corners)):
        raise ValueError('코트 모서리 4개가 필요합니다')
    corners = [[float(x), float(y)] for x, y in corners]
    court_size = str(body.get('court', '9'))
    if court_size not in va.COURTS:
        raise ValueError('court 는 9 또는 6')
    model = str(body.get('model', 's'))
    if model not in ('n', 's', 'm', 'l'):
        raise ValueError('model 은 n/s/m/l')
    fps = min(30.0, max(1.0, float(body.get('fps', 10))))
    start = max(0.0, float(body.get('start') or 0))
    end = float(body['end']) if body.get('end') else None
    job = {'id': uuid.uuid4().hex[:10], 'video': v['id'], 'status': 'queued', 'progress': 0.0, 'done': 0, 'total': 0,
           'eta': None, 'created': time.time(), 'params': {'corners': corners, 'court': court_size, 'model': model, 'fps': fps,
                                                           'start': start, 'end': end, 'tiles': bool(body.get('tiles'))},
           'cancel': False, 'message': '차례를 기다리는 중'}
    with state_lock:
        jobs[job['id']] = job
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return job


def run_job(job: dict):
    with run_lock:
        if job['cancel']:
            job.update(status='cancelled', message='취소됨')
            return
        p = job['params']
        v = videos[job['video']]
        try:
            job.update(status='running', message='모델 준비 중 (처음에는 가중치를 내려받습니다)')
            an = va.VolleyballAnalyzer(size=p['model'], tiles=p['tiles'], log=log)
            court = va.Court(p['corners'], p['court'])
            job['message'] = '분석 중'
            t0 = time.time()

            def prog(done, total, elapsed):
                rate = elapsed / max(1, done)
                job.update(done=done, total=total, progress=round(done / max(1, total), 4), eta=round(rate * (total - done)),
                           elapsed=round(elapsed))

            res = an.analyze(v['path'], court, fps=p['fps'], start=p['start'], end=p['end'], progress=prog, cancel=lambda: job['cancel'])
            res['video'].update(id=v['id'], title=v.get('title', ''))
            RES_DIR.mkdir(parents=True, exist_ok=True)
            out = RES_DIR / f"{job['id']}.json"
            out.write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
            va.save_excel(res, str(RES_DIR / f"{job['id']}.xlsx"))
            job.update(status='cancelled' if job['cancel'] else 'done', progress=1.0 if not job['cancel'] else job['progress'],
                       message=f"터치 {len(res['events'])}개 찾음 · {round(time.time() - t0)}초", result=res)
            log('분석 끝', job['id'], job['message'])
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            job.update(status='error', message=str(e)[:500])


def job_public(job: dict, with_result: bool) -> dict:
    out = {k: v for k, v in job.items() if k not in ('result', 'cancel')}
    if with_result and 'result' in job:
        out['result'] = job['result']
    elif not with_result and 'result' in job:
        out['summary'] = job['result']['stats']
    return out


# ---------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = 'VollyballAnalyzer/' + VERSION

    def log_message(self, fmt, *args):
        pass

    def _origin_ok(self) -> bool:
        o = self.headers.get('Origin')
        return o is None or bool(ALLOWED_ORIGIN.match(o))

    def _host_ok(self) -> bool:                    # 다른 사이트가 DNS 를 바꿔 이 서버에 붙는 것(리바인딩) 막기
        host = (self.headers.get('Host') or '').split(':')[0]
        return host in ('127.0.0.1', 'localhost')

    def _cors(self):
        o = self.headers.get('Origin')
        if o and ALLOWED_ORIGIN.match(o):
            self.send_header('Access-Control-Allow-Origin', o)
            self.send_header('Vary', 'Origin')

    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None):
        try:
            self.send_response(status)
            self._cors()
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            pass                                    # 브라우저가 기다리다 먼저 끊은 경우 — 무시

    def _json(self, obj, status: int = 200):
        self._send(status, json.dumps(obj, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')

    def _err(self, status: int, msg: str):
        self._json({'error': msg}, status)

    def _body(self) -> dict:
        n = int(self.headers.get('Content-Length') or 0)
        if n > 1_000_000:
            raise ValueError('요청이 너무 큽니다')
        raw = self.rfile.read(n) if n else b'{}'
        return json.loads(raw.decode('utf-8') or '{}')

    def do_OPTIONS(self):
        if not self._origin_ok() or not self._host_ok():
            return self._err(403, 'forbidden')
        self.send_response(204)
        self._cors()
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        if self.headers.get('Access-Control-Request-Private-Network'):
            self.send_header('Access-Control-Allow-Private-Network', 'true')   # 공개 사이트 → 이 PC 접근 허용(크롬)
        self.send_header('Access-Control-Max-Age', '600')
        self.end_headers()

    def do_GET(self):
        if not self._origin_ok() or not self._host_ok():
            return self._err(403, 'forbidden')
        u = urlparse(self.path)
        parts = [p for p in u.path.split('/') if p]
        q = parse_qs(u.query)
        try:
            if not parts:
                html = (f'<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>배구 영상 인식 서버</title>'
                        f'<body style="font-family:sans-serif;padding:24px;line-height:1.6;background:#0b1d2a;color:#eef6fb">'
                        f'<h2>🏐 배구 영상 인식 서버가 실행 중입니다 (포트 {PORT})</h2>'
                        f'<p><a href="/app/#analysis" style="display:inline-block;padding:12px 18px;border-radius:10px;background:#ffb020;color:#2a1a00;'
                        f'font-weight:800;text-decoration:none">▶ 경기 분석 화면 열기 (이 PC 서버에서)</a></p>'
                        f'<p>이 주소로 열면 화면과 분석 서버가 같은 주소라 브라우저 권한 없이 「🤖 AI 영상 인식」이 바로 연결됩니다.<br>'
                        f'<a href="https://lodostheory.github.io/vollyball/#analysis" style="color:#ffb020">공개 사이트</a>에서도 쓸 수 있지만, '
                        f'브라우저가 공개 사이트의 PC 접근을 막으면 연결되지 않습니다.<br>서버를 실행한 창을 닫으면 서버가 꺼집니다.</p>')
                return self._send(200, html.encode('utf-8'), 'text/html; charset=utf-8')
            if parts[0] == 'app':
                if u.path == '/app':
                    return self._send(301, b'', 'text/plain', {'Location': '/app/'})
                name = parts[1] if len(parts) > 1 else 'index.html'
                f = SITE / name
                if len(parts) > 2 or not re.fullmatch(r'[\w.-]+', name) or f.suffix.lower() not in SITE_TYPES or not f.is_file():
                    return self._err(404, 'not_found')
                return self._send(200, f.read_bytes(), SITE_TYPES[f.suffix.lower()])
            if parts == ['health']:
                busy = next((j['id'] for j in jobs.values() if j['status'] in ('running', 'queued')), None)
                return self._json({'ok': True, 'version': VERSION, 'device': DEVICE, 'gpu': GPU, 'busy': busy,
                                   'ball_model': Path(va.default_ball_model()).name})
            if parts == ['videos']:
                scan_videos()
                return self._json({'videos': [{k: v[k] for k in v if k != 'path'} for v in videos.values()]})
            if len(parts) >= 2 and parts[0] == 'videos':
                scan_videos()
                v = videos.get(parts[1])
                if not v:
                    return self._err(404, '영상이 없습니다')
                if len(parts) == 2:
                    return self._json({k: v[k] for k in v if k != 'path'})
                if parts[2:] == ['frame'] and v['status'] == 'ready':
                    t = float(q.get('t', ['0'])[0])
                    w = int(q.get('w', ['1280'])[0])
                    return self._send(200, frame_jpeg(v, t, min(1920, max(320, w))), 'image/jpeg')
                return self._err(404, 'not_found')
            if parts == ['jobs']:
                return self._json({'jobs': [job_public(j, False) for j in sorted(jobs.values(), key=lambda j: -j['created'])]})
            if len(parts) >= 2 and parts[0] == 'jobs':
                job = jobs.get(parts[1])
                if not job:
                    f = RES_DIR / f'{parts[1]}.json'
                    if re.fullmatch(r'[0-9a-f]{10}', parts[1]) and f.exists():   # 서버를 다시 켠 뒤에도 지난 결과를 돌려준다
                        res = json.loads(f.read_text(encoding='utf-8'))
                        job = jobs[parts[1]] = {'id': parts[1], 'status': 'done', 'progress': 1.0, 'created': f.stat().st_mtime,
                                                'video': res['video'].get('id'), 'message': f"터치 {len(res['events'])}개", 'result': res}
                    else:
                        return self._err(404, '작업이 없습니다')
                if len(parts) == 2:
                    return self._json(job_public(job, True))
                if parts[2:] == ['excel']:
                    f = RES_DIR / f"{job['id']}.xlsx"
                    if not f.exists():
                        return self._err(404, '엑셀이 아직 없습니다')
                    return self._send(200, f.read_bytes(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                      {'Content-Disposition': f'attachment; filename="volleyball_{job["id"]}.xlsx"'})
            return self._err(404, 'not_found')
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._err(500, str(e)[:300])

    def do_POST(self):
        if not self._origin_ok() or not self._host_ok():
            return self._err(403, 'forbidden')
        parts = [p for p in urlparse(self.path).path.split('/') if p]
        try:
            body = self._body()
            if parts == ['videos']:
                v = start_download(str(body.get('url', '')))
                return self._json({k: v[k] for k in v if k != 'path'})
            if parts == ['jobs']:
                return self._json(job_public(start_job(body), False))
            if len(parts) == 3 and parts[0] == 'jobs' and parts[2] == 'cancel':
                job = jobs.get(parts[1])
                if not job:
                    return self._err(404, '작업이 없습니다')
                job['cancel'] = True
                return self._json({'ok': True})
            return self._err(404, 'not_found')
        except ValueError as e:
            return self._err(400, str(e))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._err(500, str(e)[:300])


def main():
    global PORT, DEVICE, GPU
    for s in (sys.stdout, sys.stderr):              # 콘솔·파일 어디로 출력해도 한글이 깨지거나 멈추지 않게
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description='배구 영상 인식 로컬 서버')
    ap.add_argument('--port', type=int, default=8765)
    a = ap.parse_args()
    PORT = a.port
    for d in (VID_DIR, RES_DIR, LOCAL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    scan_videos()
    DEVICE = va.pick_device()
    if DEVICE.startswith('cuda'):
        import torch
        GPU = torch.cuda.get_device_name(0)
    srv = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    print('=' * 64)
    print(f' 🏐 배구 영상 인식 서버  http://127.0.0.1:{PORT}   (장치: {GPU or DEVICE})')
    print(' 사이트 「📹 경기 분석」 탭 → 「🤖 AI 영상 인식」에서 사용하세요.')
    print(' 이 창을 닫으면 서버가 꺼집니다. 영상은 data\\videos, 결과는 data\\results 에 저장됩니다.')
    print('=' * 64, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
