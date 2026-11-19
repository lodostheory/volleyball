"""배구 경기 영상 인식 파이프라인 (VOLLYBALL MANIA 분석 서버의 핵심)

  ① 코트 네 모서리 → 실제 코트(m) 원근 변환(Homography), 코트 밖 관중·대기 선수 가리기(ROI 마스크)
  ② YOLO 포즈(17관절) + ByteTrack 으로 선수마다 track_id 를 유지
  ③ 배구공 탐지: COCO 'sports ball' 또는 배구공으로 추가 학습한 models/volleyball_ball.pt,
     멀리 있는 작은 공은 화면을 2×2 로 잘라서 다시 찾기(슬라이싱) 선택 가능
  ④ 공 궤적이 꺾이거나 갑자기 빨라지는 순간 가장 가까운 선수를 터치한 선수로 보고,
     관절 자세(점프·손 위치·낮은 자세) + 코트 위치(엔드라인·네트) + 랠리 흐름으로
     서브(S) · 리시브(R) · 토스(E) · 공격(A) · 블로킹(B) · 디그(D) 를 규칙으로 판별
  ⑤ (시간, 트랙, 동작, 코트 위치, 그 순간 모든 선수 위치) 이벤트를 JSON · 엑셀로 저장

명령줄 사용 예:
  python volleyball_analyzer.py --video gym.mp4                     # 첫 화면에서 코트 모서리를 마우스로 찍음
  python volleyball_analyzer.py --video gym.mp4 --corners "420,280 1500,280 1780,950 140,950" --excel stats.xlsx
  python volleyball_analyzer.py --youtube https://youtu.be/XXXXXXXXXXX --start 60 --end 600 --fps 10
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = HERE / 'models'
COURTS = {'9': (10.5, 21.0), '6': (9.0, 18.0)}            # 규격별 (너비, 길이) m — 9인제 / 6인제
SKILL_KO = {'S': '서브', 'R': '리시브', 'E': '토스', 'A': '공격', 'B': '블로킹', 'D': '디그'}
# COCO 17 관절 번호
NOSE, LSH, RSH, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN = 0, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16
KP_THR = 0.35                                              # 이보다 신뢰도가 낮은 관절은 쓰지 않는다(가려진 관절은 좌표 0,0)


def pick_device(pref: str | None = None) -> str:
    if pref:
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda:0'
    except Exception:
        pass
    return 'cpu'


def weights(name: str) -> str:
    """가중치 경로. models/ 에 없으면 ultralytics 가 처음 쓸 때 그 자리로 내려받는다."""
    p = Path(name)
    if p.is_absolute() or p.exists():
        return str(p)
    MODELS.mkdir(exist_ok=True)
    return str(MODELS / name)


def default_ball_model(size: str = 's') -> str:
    custom = MODELS / 'volleyball_ball.pt'                 # 배구공으로 추가 학습한 가중치가 있으면 우선 사용
    return str(custom) if custom.exists() else weights(f'yolo11{size}.pt')


def fmt_t(t: float) -> str:
    t = int(round(t))
    return f'{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}' if t >= 3600 else f'{t // 60}:{t % 60:02d}'


# ---------------------------------------------------------------------------------------------
# ① 코트
# ---------------------------------------------------------------------------------------------
class Court:
    """영상 속 코트 바닥 네 모서리(먼 쪽 왼쪽 → 먼 쪽 오른쪽 → 가까운 쪽 오른쪽 → 가까운 쪽 왼쪽)로
    실제 코트 좌표(m)를 만든다. x: 0~너비(왼→오), y: 0(먼 쪽 엔드라인) ~ 길이(가까운 쪽 엔드라인), 네트 = 길이/2"""

    def __init__(self, corners, size: str = '9'):
        if size not in COURTS:
            raise ValueError('court size must be "9" or "6"')
        if len(corners) != 4:
            raise ValueError('need 4 corners')
        self.size = size
        self.W, self.L = COURTS[size]
        self.net = self.L / 2
        self.corners = [[float(x), float(y)] for x, y in corners]
        src = np.array(self.corners, np.float32)
        dst = np.array([[0, 0], [self.W, 0], [self.W, self.L], [0, self.L]], np.float32)
        self.H = cv2.getPerspectiveTransform(src, dst)
        self.Hi = np.linalg.inv(self.H)
        # 동차좌표 w 의 부호는 행렬 배율에 따라 뒤집힐 수 있으므로, 코트 가운데(카메라 앞이 확실한 점)를 기준으로 삼는다
        self._ws = 1.0 if (self.Hi @ np.array([self.W / 2, self.L / 2, 1.0]))[2] > 0 else -1.0

    def to_court(self, pts) -> np.ndarray:
        a = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(a, self.H).reshape(-1, 2)

    def to_image(self, pts) -> np.ndarray:
        a = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(a, self.Hi).reshape(-1, 2)

    def _project_ok(self, x, y):
        """코트 점 → 영상 점. 카메라 뒤쪽으로 넘어가는 점이면 False"""
        v = self.Hi @ np.array([x, y, 1.0])
        return v[2] * self._ws > 1e-9, (v[:2] / v[2]) if abs(v[2]) > 1e-12 else None

    def inside(self, x, y, side_m=1.5, end_m=4.5) -> bool:
        return -side_m <= x <= self.W + side_m and -end_m <= y <= self.L + end_m

    def roi_mask(self, w, h, side_m=1.5, end_m=4.5, up_m=3.6) -> np.ndarray:
        """코트(+여유) 바닥과 그 위로 선수 키(점프 포함)만큼 올린 영역만 남기는 마스크 — 관중석·벤치를 가린다"""
        pts = []
        floor = [[-side_m, -end_m], [self.W + side_m, -end_m], [self.W + side_m, self.L + end_m], [-side_m, self.L + end_m]]
        for cx, cy in floor + [[0, 0], [self.W, 0], [self.W, self.L], [0, self.L]]:
            ok, p = self._project_ok(cx, cy)
            if not ok or p is None or not np.all(np.isfinite(p)) or abs(p[0]) > 4 * w or abs(p[1]) > 4 * h:
                continue                                    # 카메라 뒤쪽으로 넘어가는 점은 버린다
            ok2, q = self._project_ok(cx + 1, cy)          # 그 자리에서 1m 가 몇 픽셀인지
            s = float(np.hypot(*(q - p))) if ok2 and q is not None else 0
            pts.append(p)
            pts.append(np.array([p[0], p[1] - up_m * s]))
        for x, y in self.corners[2:]:                       # 가까운 쪽 엔드라인 아래(카메라 쪽)는 화면 끝까지
            pts.append(np.array([x, h + 5.0]))
        hull = cv2.convexHull(np.array(pts, np.float32)).astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        return mask

    def draw(self, img, color=(255, 255, 255)):
        L, W = self.L, self.W
        lines = [[(0, 0), (W, 0)], [(W, 0), (W, L)], [(W, L), (0, L)], [(0, L), (0, 0)], [(0, L / 2), (W, L / 2)]]
        for a, b in lines:
            p, q = self.to_image([a, b]).astype(int)
            cv2.line(img, tuple(p), tuple(q), color, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------------------------
# ② 선수 (포즈 + 추적)
# ---------------------------------------------------------------------------------------------
class Player:
    __slots__ = ('tid', 'box', 'kp', 'kc', 'h', 'cx', 'cy', 'side', 'jump', 'foot_y')

    def ys(self, idxs):
        return [float(self.kp[i][1]) for i in idxs if self.kc[i] >= KP_THR]

    def pts(self, idxs):
        return [self.kp[i] for i in idxs if self.kc[i] >= KP_THR]


def _mean(v):
    return sum(v) / len(v) if v else None


# ---------------------------------------------------------------------------------------------
# ③ 공
# ---------------------------------------------------------------------------------------------
class BallDetector:
    def __init__(self, path: str, device: str, imgsz: int = 1280, conf: float = 0.15, tiles: bool = False, half: bool = False):
        from ultralytics import YOLO
        self.m = YOLO(path)
        names = self.m.names if isinstance(self.m.names, dict) else dict(enumerate(self.m.names))
        ids = [int(k) for k, v in names.items() if 'ball' in str(v).lower()]
        self.cls = ids or None                            # COCO 면 32(sports ball), 배구공 전용이면 그 클래스
        self.device, self.imgsz, self.conf, self.tiles, self.half = device, imgsz, conf, tiles, half

    @staticmethod
    def _boxes(r, ox=0, oy=0):
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        xyxy = b.xyxy.cpu().numpy()
        cf = b.conf.cpu().numpy()
        return [((x1 + x2) / 2 + ox, (y1 + y2) / 2 + oy, float(c), float(max(x2 - x1, y2 - y1))) for (x1, y1, x2, y2), c in zip(xyxy, cf)]

    def detect(self, img):
        kw = dict(classes=self.cls, conf=self.conf, device=self.device, verbose=False)
        if not self.tiles:
            return self._boxes(self.m.predict(img, imgsz=self.imgsz, **kw)[0])
        # 슬라이싱: 겹치게 2×2 로 잘라 각각 찾으면 먼 쪽의 작은 공도 잡힌다 (느려짐)
        H, W = img.shape[:2]
        tw, th = int(W * 0.56), int(H * 0.56)
        offs = [(ox, oy) for oy in (0, H - th) for ox in (0, W - tw)]
        crops = [img[oy:oy + th, ox:ox + tw] for ox, oy in offs]
        out = []
        for r, (ox, oy) in zip(self.m.predict(crops, imgsz=min(self.imgsz, 960), **kw), offs):
            out += self._boxes(r, ox, oy)
        out.sort(key=lambda b: -b[2])
        merged = []
        for b in out:                                      # 겹친 조각에서 두 번 잡힌 공은 하나로
            if all(math.hypot(b[0] - m[0], b[1] - m[1]) > max(b[3], m[3]) for m in merged):
                merged.append(b)
        return merged


# ---------------------------------------------------------------------------------------------
# ④ 분석기
# ---------------------------------------------------------------------------------------------
class VolleyballAnalyzer:
    def __init__(self, pose_model: str | None = None, ball_model: str | None = None, size: str = 's', device: str | None = None,
                 imgsz: int = 1280, ball_imgsz: int = 1280, tiles: bool = False, use_mask: bool = True, log=print):
        # imgsz 1280: 네트 너머 작은 선수까지 잡힌다 (960 보다 먼 쪽 선수 2~3배, 시간은 GTX1650 기준 +0.02초)
        from ultralytics import YOLO
        self.device = pick_device(device)
        self.half = False          # FP16 은 GTX 16xx 계열에서 결과가 비어 버리고(NaN) 더 느려서 쓰지 않는다
        self.log = log
        self.pose_path = pose_model or weights(f'yolo11{size}-pose.pt')
        self.ball_path = ball_model or default_ball_model(size)
        self.log(f'모델 준비: 포즈 {Path(self.pose_path).name}, 공 {Path(self.ball_path).name}, 장치 {self.device}')
        self.pose = YOLO(self.pose_path)
        self.ball = BallDetector(self.ball_path, self.device, ball_imgsz, tiles=tiles, half=self.half)
        self.imgsz = imgsz
        self.use_mask = use_mask
        self.tracker = self._tracker_cfg()

    @staticmethod
    def _tracker_cfg() -> str:
        """ByteTrack 기본 설정을 배구에 맞게 조금 느슨하게: 선수가 겹쳤다 나와도 같은 번호를 오래 유지"""
        try:
            import yaml
            from ultralytics.utils import ROOT
            src = ROOT / 'cfg' / 'trackers' / 'bytetrack.yaml'
            cfg = yaml.safe_load(src.read_text(encoding='utf-8'))
            cfg.update(track_buffer=90, match_thresh=0.9, new_track_thresh=0.3)
            dst = HERE / 'data' / 'bytetrack_volley.yaml'
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
            return str(dst)
        except Exception:
            return 'bytetrack.yaml'

    # ---- 선수 -------------------------------------------------------------------------------
    def _players(self, img, court: Court, foot_hist) -> list[Player]:
        r = self.pose.track(img, persist=True, tracker=self.tracker, classes=[0], conf=0.2, iou=0.6, imgsz=self.imgsz,
                            device=self.device, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None or r.keypoints is None:
            return []
        ids = r.boxes.id.int().cpu().tolist()
        boxes = r.boxes.xyxy.cpu().numpy()
        kps = r.keypoints.xy.cpu().numpy()
        kcs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kps.shape[:2])
        out = []
        for tid, box, kp, kc in zip(ids, boxes, kps, kcs):
            p = Player()
            p.tid, p.box, p.kp, p.kc = int(tid), box, kp, kc
            p.h = float(max(1.0, box[3] - box[1]))
            ankles = p.pts([LAN, RAN])
            foot_x = float(_mean([a[0] for a in ankles])) if ankles else float((box[0] + box[2]) / 2)
            p.foot_y = float(max(a[1] for a in ankles)) if ankles else float(box[3])
            # 점프: 최근 2초 동안의 발 높이(바닥) 기준으로 발이 몸 높이의 10% 넘게 떠 있으면
            hist = foot_hist[p.tid]
            hist.append(p.foot_y)
            ground = float(np.percentile(hist, 80)) if len(hist) >= 3 else p.foot_y
            p.jump = (ground - p.foot_y) / p.h > 0.10
            # 코트 위치는 바닥(점프 중이면 바닥 높이) 기준 발 위치로
            cx, cy = court.to_court([[foot_x, max(p.foot_y, ground)]])[0]
            if not court.inside(cx, cy):
                continue                                    # 코트 밖 사람(관중·심판·벤치)
            p.cx, p.cy = float(cx), float(cy)
            p.side = 'far' if p.cy < court.net else 'near'
            out.append(p)
        return out

    # ---- 동작 판별 -----------------------------------------------------------------------------
    def _classify(self, p: Player, ball, court: Court, t: float, accel: bool):
        ctx = self.ctx
        if ctx['t'] is None or t - ctx['t'] > 4.0:           # 4초 넘게 터치가 없으면 새 랠리
            ctx.update(side=None, sk=None, n=0)
        sh = _mean(p.ys([LSH, RSH]))
        hip = _mean(p.ys([LHIP, RHIP]))
        knee = _mean(p.ys([LKN, RKN]))
        wrists = p.ys([LWR, RWR])
        top = float(p.box[1])
        sh = sh if sh is not None else top + 0.22 * p.h
        hip = hip if hip is not None else top + 0.52 * p.h
        torso = max(1.0, hip - sh)
        wrist = min(wrists) if wrists else None
        hands_up = wrist is not None and wrist < sh - 0.03 * p.h
        both_up = len(wrists) == 2 and max(wrists) < sh
        ball_high = ball[1] < sh
        knee_bend = (knee - hip) / torso if knee is not None else 1.0
        low = knee_bend < 0.75 or ball[1] > hip
        near_net = abs(p.cy - court.net) < 1.3
        end_dist = min(abs(p.cy), abs(p.cy - court.L))
        behind_end = p.cy < -0.3 or p.cy > court.L + 0.3
        same_side = ctx['side'] == p.side
        n_side = ctx['n'] + 1 if same_side else 1            # 이 쪽 코트에서 몇 번째 터치인지
        other_hit = ctx['side'] is not None and not same_side and ctx['t'] is not None and t - ctx['t'] < 1.6

        if behind_end or (ctx['side'] is None and end_dist < 1.2 and hands_up and ball_high):
            sk, c = 'S', 0.8 if behind_end else 0.55
        elif near_net and (both_up or (hands_up and p.jump)) and other_hit and ctx['sk'] in ('A', 'E', None):
            sk, c = 'B', 0.7
        elif ctx['sk'] == 'S' and not same_side:
            sk, c = 'R', 0.75
        elif hands_up and ball_high and (p.jump or accel or n_side >= 3):
            sk, c = 'A', 0.7 if p.jump else 0.5
        elif hands_up and ball_high:
            sk, c = 'E', 0.6 if n_side == 2 else 0.45
        elif low or not ball_high:
            sk, c = 'D', 0.55
        else:
            sk, c = ('E', 0.35) if n_side == 2 else ('A', 0.35) if n_side >= 3 else ('D', 0.35)
        ctx.update(side=p.side, sk=sk, t=t, n=n_side)
        return sk, c, dict(jump=bool(p.jump), hands=bool(hands_up), low=bool(low), near_net=bool(near_net))

    # ---- 터치 찾기 -----------------------------------------------------------------------------
    @staticmethod
    def _nearest(ball, players):
        """공과 가장 가까운 선수 (손목까지 거리, 없으면 상체 박스까지 거리를 키로 나눈 값)"""
        best, bd = None, 1e9
        for p in players:
            ws = p.pts([LWR, RWR])
            if ws:
                d = min(math.hypot(ball[0] - w[0], ball[1] - w[1]) for w in ws)
            else:
                x1, y1, x2, y2 = p.box
                dx = max(x1 - ball[0], 0, ball[0] - x2)
                dy = max(y1 - ball[1], 0, ball[1] - (y1 + y2) / 2)
                d = math.hypot(dx, dy) + 0.1 * p.h
            d /= p.h
            if d < bd:
                best, bd = p, d
        return best, bd

    def _event(self, k_obs, court):
        """공 관측점 k_obs(=(k, t, frame, x, y, players)) 에서 터치가 있었다고 보고 이벤트를 만든다"""
        k, t, fi, bx, by, players, accel = k_obs
        p, d = self._nearest((bx, by), players)
        if p is None or d > 0.6:
            return None
        if self.events:
            last = self.events[-1]
            if t - last['t'] < 0.3 or (last['track'] == p.tid and t - last['t'] < 0.6):
                return None                                 # 같은 터치를 두 번 세지 않게
        sk, conf, feats = self._classify(p, (bx, by), court, t, accel)
        conf = round(conf * (1.0 if d < 0.3 else 0.8), 2)
        return {
            't': round(t, 2), 'frame': int(fi), 'track': p.tid, 'side': p.side, 'sk': sk, 'conf': conf,
            'x': round(p.cx, 2), 'y': round(p.cy, 2), 'ball': [int(bx), int(by)], **feats,
            'players': [[q.tid, q.side, round(q.cx, 2), round(q.cy, 2)] for q in players],
        }

    def _check_turn(self, court, h):
        """최근 공 관측 3개로, 가운데 점에서 공이 꺾였는지(방향 30°↑, 내려오다 솟음, 1.8배 가속) 본다"""
        if len(self.obs) < 3:
            return
        a, b, c = self.obs[-3], self.obs[-2], self.obs[-1]
        if c[0] - b[0] > 2 or b[0] - a[0] > 2:
            return                                          # 중간에 공을 놓친 구간은 판단하지 않는다
        v1 = ((b[3] - a[3]) / (b[0] - a[0]), (b[4] - a[4]) / (b[0] - a[0]))
        v2 = ((c[3] - b[3]) / (c[0] - b[0]), (c[4] - b[4]) / (c[0] - b[0]))
        s1, s2 = math.hypot(*v1), math.hypot(*v2)
        mv = 0.004 * h * self.step_scale
        turn = accel = False
        if s1 > mv and s2 > mv:
            cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (s1 * s2)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
            up_after_down = v1[1] > 0.5 * mv and v2[1] < -0.5 * mv
            accel = s2 > 1.8 * s1
            turn = ang > 30 or up_after_down or accel
        elif s1 <= mv and s2 > 3 * mv:                      # 멈춰 있던(들고 있던) 공이 날아감 = 서브
            turn = accel = True
        if turn:
            ev = self._event((*b[:6], accel), court)
            if ev:
                self.events.append(ev)
                self.log(f"  {fmt_t(ev['t'])} {ev['side']:>4} T{ev['track']:<3} {SKILL_KO[ev['sk']]}")

    def _refine(self):
        """랠리 흐름으로 판별을 다듬는다: 서브 뒤에 같은 쪽 터치가 바로 오면 서브가 아니고,
        리시브는 바로 앞이 상대 서브일 때만 리시브다"""
        ev = self.events

        def other(e):
            return 'A' if e.get('jump') and e.get('hands') else 'E' if e.get('hands') else 'D'
        for i, e in enumerate(ev):
            prv = ev[i - 1] if i else None
            nxt = ev[i + 1] if i + 1 < len(ev) else None
            if e['sk'] != 'S':
                continue
            if (nxt and nxt['side'] == e['side'] and nxt['t'] - e['t'] < 2.5) or (prv and e['t'] - prv['t'] < 3.0):
                e['sk'], e['conf'], e['fixed'] = other(e), round(e['conf'] * 0.7, 2), 'S'
        for i, e in enumerate(ev):
            prv = ev[i - 1] if i else None
            if e['sk'] == 'R' and not (prv and prv['sk'] == 'S' and prv['side'] != e['side'] and e['t'] - prv['t'] < 4.0):
                e['sk'], e['fixed'] = 'D', 'R'
            elif e['sk'] == 'D' and prv and prv['sk'] == 'S' and prv['side'] != e['side'] and e['t'] - prv['t'] < 4.0:
                e['sk'], e['fixed'] = 'R', 'D'      # 상대 서브 바로 다음 터치는 리시브

    # ---- 전체 ---------------------------------------------------------------------------------
    def analyze(self, video_path, court: Court, fps: float = 10.0, start: float = 0.0, end: float | None = None,
                progress=None, cancel=None, annotate: str | None = None) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f'영상을 열 수 없습니다: {video_path}')
        vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step = max(1, round(vfps / max(0.5, fps)))
        self.step_scale = step / max(1.0, vfps / 10)         # 공 이동량 기준을 분석 간격에 맞춤
        f0 = max(0, int(start * vfps))
        f1 = min(n_total, int(end * vfps)) if end else n_total
        if f0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        mask = court.roi_mask(w, h) if self.use_mask else None
        if mask is not None and mask.mean() / 255 < 0.08:
            self.log('코트 가리기 영역이 너무 작아 가리기를 끕니다 (코트 모서리를 확인하세요)')
            mask = None
        foot_hist = defaultdict(lambda: deque(maxlen=max(3, int(2 * vfps / step))))
        self.events, self.obs, self.ctx = [], deque(maxlen=6), dict(side=None, sk=None, t=None, n=0)
        tracks = defaultdict(lambda: {'frames': 0, 'near': 0, 'far': 0, 'x': 0.0, 'y': 0.0})
        writer = None
        if annotate:
            writer = cv2.VideoWriter(annotate, cv2.VideoWriter_fourcc(*'mp4v'), vfps / step, (w, h))
        if hasattr(self.pose, 'predictor') and self.pose.predictor is not None:
            self.pose.predictor = None                       # 이전 영상의 추적 번호를 이어 쓰지 않게
        t_start, k, fi, ball_seen = time.time(), 0, f0, 0
        n_steps = max(1, (f1 - f0) // step)
        try:
            while fi < f1:
                if cancel and cancel():
                    break
                if (fi - f0) % step:
                    if not cap.grab():
                        break
                    fi += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                t = fi / vfps
                src = cv2.bitwise_and(frame, frame, mask=mask) if mask is not None else frame
                players = self._players(src, court, foot_hist)
                for p in players:
                    tr = tracks[p.tid]
                    tr['frames'] += 1
                    tr[p.side] += 1
                    tr['x'] += p.cx
                    tr['y'] += p.cy
                # 공: 직전 위치에서 예상되는 자리와 가장 가까운 후보(없으면 가장 확실한 후보)
                cands = [b for b in self.ball.detect(src) if b[3] < 0.08 * h]
                ball = None
                if cands:
                    if self.obs and k - self.obs[-1][0] <= 3:
                        lk, _, _, lx, ly, _ = self.obs[-1]
                        if len(self.obs) >= 2 and lk - self.obs[-2][0] <= 2:
                            pk, _, _, px, py, _ = self.obs[-2]
                            vx, vy = (lx - px) / (lk - pk), (ly - py) / (lk - pk)
                        else:
                            vx = vy = 0.0
                        ex, ey = lx + vx * (k - lk), ly + vy * (k - lk)
                        gate = max(0.12 * h, 2.0 * math.hypot(vx, vy) * (k - lk))
                        near = [b for b in cands if math.hypot(b[0] - ex, b[1] - ey) < gate]
                        ball = min(near, key=lambda b: math.hypot(b[0] - ex, b[1] - ey)) if near else None
                    if ball is None:
                        ball = max(cands, key=lambda b: b[2])
                if ball is not None:
                    ball_seen += 1
                    self.obs.append((k, t, fi, ball[0], ball[1], players))
                    self._check_turn(court, h)
                if writer is not None:
                    self._draw(frame, court, players, ball, t)
                    writer.write(frame)
                k += 1
                fi += 1
                if progress and k % 5 == 0:
                    progress(k, n_steps, time.time() - t_start)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        elapsed = time.time() - t_start
        self._refine()
        if progress:
            progress(k, n_steps, elapsed)
        track_list = {}
        for tid, tr in tracks.items():
            if tr['frames'] < 3:
                continue
            track_list[str(tid)] = {'frames': tr['frames'], 'side': 'near' if tr['near'] >= tr['far'] else 'far',
                                    'x': round(tr['x'] / tr['frames'], 2), 'y': round(tr['y'] / tr['frames'], 2),
                                    'touches': sum(1 for e in self.events if e['track'] == tid)}
        return {
            'version': 1,
            'video': {'path': str(video_path), 'fps': vfps, 'width': w, 'height': h, 'frames': n_total, 'duration': round(n_total / vfps, 2)},
            'court': {'size': court.size, 'W': court.W, 'L': court.L, 'corners': court.corners},
            'params': {'fps': round(vfps / step, 2), 'start': round(f0 / vfps, 2), 'end': round(min(fi, f1) / vfps, 2),
                       'pose': Path(self.pose_path).name, 'ball': Path(self.ball_path).name, 'device': self.device,
                       'tiles': self.ball.tiles, 'mask': self.use_mask},
            'stats': {'frames_analyzed': k, 'ball_frames': ball_seen, 'elapsed_sec': round(elapsed, 1),
                      'sec_per_frame': round(elapsed / max(1, k), 3), 'events': len(self.events),
                      'by_skill': dict(Counter(e['sk'] for e in self.events))},
            'events': self.events,
            'tracks': track_list,
        }

    @staticmethod
    def _draw(img, court, players, ball, t):
        court.draw(img, (255, 255, 255))
        for p in players:
            col = (240, 200, 80) if p.side == 'near' else (90, 90, 255)
            x1, y1, x2, y2 = p.box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, f'T{p.tid}{" J" if p.jump else ""}', (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if ball is not None:
            cv2.circle(img, (int(ball[0]), int(ball[1])), max(6, int(ball[3] / 2)), (0, 255, 255), 2)
        cv2.putText(img, fmt_t(t), (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


# ---------------------------------------------------------------------------------------------
# ⑤ 엑셀 리포트
# ---------------------------------------------------------------------------------------------
def save_excel(result: dict, path: str) -> str:
    import pandas as pd
    side_ko = {'near': '가까운 쪽', 'far': '먼 쪽'}
    rows = [{
        '시간(초)': e['t'], '시각': fmt_t(e['t']), '프레임': e['frame'], '선수_트랙': f"T{e['track']}", '코트_쪽': side_ko[e['side']],
        '동작': SKILL_KO[e['sk']], '코드': e['sk'], '신뢰도': e['conf'], '코트위치_X(m)': e['x'], '코트위치_Y(m)': e['y'],
        '점프': 'O' if e.get('jump') else '', '손_위': 'O' if e.get('hands') else '', '낮은_자세': 'O' if e.get('low') else '',
    } for e in result['events']]
    cols = ['시간(초)', '시각', '프레임', '선수_트랙', '코트_쪽', '동작', '코드', '신뢰도', '코트위치_X(m)', '코트위치_Y(m)', '점프', '손_위', '낮은_자세']
    df = pd.DataFrame(rows, columns=cols)
    info = pd.DataFrame([
        ['영상', result['video']['path']], ['코트 규격', f"{result['court']['size']}인제 {result['court']['L']}×{result['court']['W']}m"],
        ['분석 구간', f"{fmt_t(result['params']['start'])} ~ {fmt_t(result['params']['end'])}"], ['분석 프레임', f"초당 {result['params']['fps']}장"],
        ['모델', f"{result['params']['pose']} / {result['params']['ball']} ({result['params']['device']})"],
        ['찾은 터치', len(result['events'])], ['공이 보인 프레임', f"{result['stats']['ball_frames']} / {result['stats']['frames_analyzed']}"],
        ['주의', '규칙 기반 자동 판별 결과입니다. 트랙 번호(T)는 실제 등번호가 아니며 선수가 겹치면 바뀔 수 있습니다.'],
    ], columns=['항목', '값'])
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        if len(df):
            summary = pd.pivot_table(df, index=['코트_쪽', '선수_트랙'], columns='동작', values='프레임', aggfunc='count', fill_value=0)
            summary['합계'] = summary.sum(axis=1)
            summary.sort_values('합계', ascending=False).to_excel(xw, sheet_name='선수별_동작요약')
        df.to_excel(xw, sheet_name='전체_이벤트로그', index=False)
        info.to_excel(xw, sheet_name='분석정보', index=False)
    return path


# ---------------------------------------------------------------------------------------------
# 명령줄
# ---------------------------------------------------------------------------------------------
def pick_corners(frame) -> list[list[float]]:
    """첫 화면에서 코트 모서리 4개를 마우스로 찍는다 (먼 쪽 왼쪽 → 먼 쪽 오른쪽 → 가까운 쪽 오른쪽 → 가까운 쪽 왼쪽)"""
    pts, win = [], 'Court corners: far-left, far-right, near-right, near-left | Enter=OK Backspace=undo Esc=quit'
    scale = min(1.0, 1600 / frame.shape[1])
    view = cv2.resize(frame, None, fx=scale, fy=scale)

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([x / scale, y / scale])

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = view.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(img, (int(x * scale), int(y * scale)), 6, (0, 255, 255), -1)
            cv2.putText(img, str(i + 1), (int(x * scale) + 8, int(y * scale) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if len(pts) > 1:
            cv2.polylines(img, [np.array([[x * scale, y * scale] for x, y in pts], np.int32)], len(pts) == 4, (0, 255, 255), 2)
        cv2.imshow(win, img)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and len(pts) == 4:
            break
        if key == 8 and pts:
            pts.pop()
        if key == 27:
            cv2.destroyAllWindows()
            sys.exit('취소했습니다')
    cv2.destroyAllWindows()
    return pts


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description='배구 경기 영상 인식 → 터치 이벤트 JSON · 엑셀')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--video', help='영상 파일 경로')
    src.add_argument('--youtube', help='유튜브 주소 (yt-dlp 로 받아서 분석)')
    ap.add_argument('--corners', help='코트 모서리 픽셀 "x,y x,y x,y x,y" (먼 쪽 왼쪽부터 시계 방향). 없으면 마우스로 찍기')
    ap.add_argument('--court', choices=['9', '6'], default='9', help='9인제(21×10.5m) / 6인제(18×9m)')
    ap.add_argument('--fps', type=float, default=10, help='초당 분석 프레임 수 (높을수록 정확·느림)')
    ap.add_argument('--start', type=float, default=0, help='분석 시작(초)')
    ap.add_argument('--end', type=float, default=None, help='분석 끝(초)')
    ap.add_argument('--model', choices=['n', 's', 'm', 'l'], default='s', help='모델 크기 (n 빠름 · s 보통 · m 정확)')
    ap.add_argument('--ball-model', help='배구공 전용 가중치 경로 (없으면 models/volleyball_ball.pt 또는 COCO)')
    ap.add_argument('--tiles', action='store_true', help='작은 공 찾기 강화(화면을 잘라서 탐지, 느림)')
    ap.add_argument('--no-mask', action='store_true', help='코트 밖 가리기 끄기')
    ap.add_argument('--device', help='cpu / cuda:0 (기본: 자동)')
    ap.add_argument('--out', default='match_events.json', help='이벤트 JSON 저장 경로')
    ap.add_argument('--excel', default='match_stats.xlsx', help='엑셀 리포트 저장 경로')
    ap.add_argument('--annotate', help='상자·공·코트를 그린 확인용 영상 저장 경로 (.mp4)')
    a = ap.parse_args()

    path = a.video
    if a.youtube:
        from youtube import download
        path, meta = download(a.youtube, HERE / 'data' / 'videos', progress=lambda f: print(f'\r내려받는 중 {f * 100:5.1f}%', end=''))
        print(f"\n받음: {meta.get('title')} ({path})")
    if a.corners:
        corners = [[float(v) for v in p.split(',')] for p in a.corners.split()]
    else:
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit('영상을 읽을 수 없습니다')
        corners = pick_corners(frame)
        print('코트 모서리:', ' '.join(f'{x:.0f},{y:.0f}' for x, y in corners))
    court = Court(corners, a.court)
    an = VolleyballAnalyzer(ball_model=a.ball_model, size=a.model, device=a.device, tiles=a.tiles, use_mask=not a.no_mask)
    res = an.analyze(path, court, fps=a.fps, start=a.start, end=a.end, annotate=a.annotate,
                     progress=lambda d, n, s: print(f'\r분석 {d}/{n} ({d / n * 100:4.1f}%) · {s:5.0f}초', end=''))
    print()
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
    save_excel(res, a.excel)
    st = res['stats']
    print(f"터치 {st['events']}개 {st['by_skill']} · 프레임당 {st['sec_per_frame']}초 → {a.out}, {a.excel}")


if __name__ == '__main__':
    main()
