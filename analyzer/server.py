"""배구 영상 인식 로컬 서버 — 사이트의 「📹 경기 분석」 탭이 이 서버를 불러 씁니다.

실행: run_server.bat 더블클릭 (처음 한 번은 라이브러리를 설치)  또는  python server.py [--port 8765]
이 PC 안(127.0.0.1)에서만 열리고, VOLLEYBALL MANIA 사이트와 이 PC 의 주소에서 온 요청만 받습니다.

API
  GET  /health                     서버·장치 상태
  POST /videos {url}               유튜브 영상 받기 시작 → {id}
  GET  /videos                     받은 영상 목록 (+ videos 폴더에 넣은 영상)
  GET  /videos/<id>                받기 진행·영상 정보
  GET  /videos/<id>/frame?t=초&w=폭  그 시각 화면(JPEG) — 코트 모서리 찍기용
  POST /videos/<id>/trim {start, end}  받은 영상의 앞뒤 잘라 내기 (유튜브 시각)
  POST /jobs {video, corners, court, fps, start, end, model, tiles}  분석 시작 → {id}
  GET  /jobs/<id>                  진행률 · 결과(이벤트)
  POST /jobs/<id>/cancel           분석 취소
  GET  /jobs/<id>/excel            엑셀 리포트 내려받기
  POST /jobs/<id>/events/delete {index, t}  후보 하나 삭제 (JSON · 엑셀 다시 저장)
  POST /jobs/clear                 끝난 분석 결과 전체 삭제 (JSON · 엑셀, 돌아가는 분석은 그대로)
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
from trim import trim_in_place  # noqa: E402
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
                           'local': bool(prefix), 'offset': float(meta.get('offset') or 0), **probe(f)}


def start_download(url: str, force: bool = False) -> dict:
    vid = youtube_id(url)
    if not vid:
        raise ValueError('유튜브 주소가 아닙니다')
    with state_lock:
        v = videos.get(vid)
        if v and v['status'] in ('downloading', 'trimming'):
            return v
        if v and v['status'] == 'ready':
            if not force:
                return v
            for f in VID_DIR.glob(vid + '.*'):          # 원본 다시 받기: 잘라 낸 파일을 지우고 처음부터
                f.unlink(missing_ok=True)
        v = videos[vid] = {'id': vid, 'status': 'downloading', 'progress': 0.0, 'title': '', 'url': url, 'offset': 0.0}

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


def start_trim(v: dict, start: float, end: float | None) -> dict:
    """받은 영상의 앞뒤를 잘라 낸다. start·end 는 유튜브(원본) 시각, 파일 안 시각은 offset 만큼 앞당겨 계산"""
    if v['status'] != 'ready':
        raise ValueError('준비된 영상이 아닙니다')
    if any(j['video'] == v['id'] and j['status'] in ('queued', 'running') for j in jobs.values()):
        raise ValueError('이 영상을 분석하는 중에는 자를 수 없습니다')
    off, dur = float(v.get('offset') or 0), float(v.get('duration') or 0)
    fs = max(0.0, start - off)
    fe = None if end is None else end - off
    if fe is not None and fe <= fs + 1:
        raise ValueError('끝 시각은 시작보다 1초 이상 뒤여야 합니다')
    if dur and fs >= dur - 1:
        raise ValueError('시작 시각이 영상 끝을 넘습니다')
    if fe is not None and dur and fe >= dur:
        fe = None
    v.update(status='trimming', progress=0.0)

    def run():
        try:
            path = Path(v['path'])
            log('영상 자르기', path.name, f'{start:.1f}초 ~ {end if end is not None else "끝"}')
            new_off, new_dur = trim_in_place(path, fs, fe, progress=lambda f: v.update(progress=round(f, 3)))
            meta_f = path.with_suffix('.json')
            meta = json.loads(meta_f.read_text(encoding='utf-8')) if meta_f.exists() else {}
            meta['offset'] = round(off + new_off, 3)
            meta_f.write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
            v.update(status='ready', progress=1.0, offset=meta['offset'], **probe(path))
            log('영상 자르기 완료', f"원본 {meta['offset']:.1f}초부터 {v['duration']}초 남김")
        except Exception as e:  # noqa: BLE001
            v.update(status='ready', progress=1.0, error=f'자르기 실패: {e}'[:300])
            log('영상 자르기 실패', e)

    threading.Thread(target=run, daemon=True).start()
    return v


def frame_jpeg(v: dict, t: float, width: int) -> bytes:
    cap = cv2.VideoCapture(v['path'])
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t - float(v.get('offset') or 0)) * 1000)   # 유튜브 시각 → 파일 시각
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

            off = float(v.get('offset') or 0)               # 앞을 잘라 낸 영상이면 파일 0초 = 원본 off 초
            fs = max(0.0, p['start'] - off)
            fe = None if p['end'] is None else max(fs + 0.5, p['end'] - off)
            res = an.analyze(v['path'], court, fps=p['fps'], start=fs, end=fe, progress=prog, cancel=lambda: job['cancel'])
            if off:
                for e in res['events']:
                    e['t'] = round(e['t'] + off, 2)
                res['params']['start'] = round(res['params']['start'] + off, 2)
                res['params']['end'] = round(res['params']['end'] + off, 2)
            res['video'].update(id=v['id'], title=v.get('title', ''), offset=off)
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
                        f'<a href="https://lodostheory.github.io/volleyball/#analysis" style="color:#ffb020">공개 사이트</a>에서도 쓸 수 있지만, '
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
                v = start_download(str(body.get('url', '')), force=bool(body.get('force')))
                return self._json({k: v[k] for k in v if k != 'path'})
            if len(parts) == 3 and parts[0] == 'videos' and parts[2] == 'trim':
                v = videos.get(parts[1])
                if not v:
                    return self._err(404, '영상이 없습니다')
                start = float(body.get('start') or 0)
                end = float(body['end']) if body.get('end') not in (None, '') else None
                v = start_trim(v, max(0.0, start), end)
                return self._json({k: v[k] for k in v if k != 'path'})
            if parts == ['jobs']:
                return self._json(job_public(start_job(body), False))
            if parts == ['jobs', 'clear']:
                return self._json(clear_results())
            if len(parts) == 4 and parts[0] == 'jobs' and parts[2:] == ['events', 'delete']:
                return self._json(delete_event(parts[1], body))
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


def delete_event(job_id: str, body: dict) -> dict:
    """끝난 분석 결과에서 후보(터치 이벤트) 하나를 지우고 JSON · 엑셀을 다시 저장한다.
    t(시각)가 그 자리 후보와 맞을 때만 지운다 — 다른 화면에서 먼저 지워 순서가 바뀐 경우를 막음"""
    if not re.fullmatch(r'[0-9a-f]{10}', job_id):
        raise ValueError('작업 번호가 이상합니다')
    f = RES_DIR / f'{job_id}.json'
    job = jobs.get(job_id)
    if job and job['status'] in ('queued', 'running'):
        raise ValueError('분석이 끝난 뒤에 지울 수 있습니다')
    res = job.get('result') if job else None
    if res is None:
        if not f.exists():
            raise ValueError('분석 결과가 없습니다')
        res = json.loads(f.read_text(encoding='utf-8'))
    evs = res.get('events') or []
    i = int(body.get('index', -1))
    if not 0 <= i < len(evs):
        raise ValueError('그런 후보가 없습니다')
    if body.get('t') is not None and abs(float(evs[i].get('t', 0)) - float(body['t'])) > 0.01:
        raise ValueError('후보 목록이 바뀌었습니다. 결과를 다시 불러오세요')
    gone = evs.pop(i)
    f.write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
    try:
        va.save_excel(res, str(RES_DIR / f'{job_id}.xlsx'))
    except Exception:  # noqa: BLE001  엑셀이 열려 있으면 못 씀 — JSON 은 이미 저장됨
        traceback.print_exc()
    if job:
        job['result'] = res
        job['message'] = f"터치 {len(evs)}개"
    log('후보 삭제', f"{job_id} #{i + 1} ({gone.get('t')}초) · 남은 후보 {len(evs)}개")
    return {'ok': True, 'events': len(evs)}


def clear_results() -> dict:
    """끝난(완료 · 취소 · 실패) 분석 결과를 모두 지운다. 돌아가거나 기다리는 분석은 남김"""
    keep = {i for i, j in jobs.items() if j['status'] in ('queued', 'running')}
    for i in [i for i in jobs if i not in keep]:
        del jobs[i]
    n = 0
    for f in RES_DIR.glob('*'):
        if f.is_file() and f.suffix.lower() in ('.json', '.xlsx') and f.stem not in keep:
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
    log('분석 결과 전체 삭제', f'파일 {n}개' + (f' · 진행 중 {len(keep)}개는 남김' if keep else ''))
    return {'deleted': n, 'kept': sorted(keep)}


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
