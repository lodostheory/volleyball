"""받은 영상의 앞뒤 자르기 (다시 압축하지 않고 영상 데이터를 그대로 옮겨 담음 → 빠르고 화질 손실 없음)

잘라 낸 파일의 0초는 원본의 `offset` 초입니다. 압축 영상은 키프레임에서만 자를 수 있어서 offset 은
요청한 시작 시각보다 조금(보통 1~4초) 앞일 수 있고, 서버는 이 offset 으로 유튜브 화면의 시각과 맞춥니다.
"""
from __future__ import annotations

import os
from pathlib import Path


def trim_copy(src: Path, dst: Path, start: float, end: float | None, progress=None) -> tuple[float, float]:
    """src 의 [start, end] 구간을 dst 로. 돌려주는 값: (offset = 원본 기준 새 파일의 0초, 새 파일 길이 초)"""
    import av
    inp = av.open(str(src))
    try:
        ins = inp.streams.video[0]
        tb = ins.time_base
        out = av.open(str(dst), 'w', format='mp4')
        try:
            outs = out.add_stream_from_template(ins)
            if start > 0:                                   # 시작 시각 바로 앞 키프레임으로
                inp.seek(int(start / tb), stream=ins, backward=True, any_frame=False)
            base = None
            last_t = 0.0
            span = (end if end else float(ins.duration * tb) if ins.duration else 0) - start
            for pkt in inp.demux(ins):
                if pkt.dts is None or pkt.pts is None:
                    continue
                if base is None:
                    base = min(pkt.pts, pkt.dts)
                t = float(pkt.dts * tb)
                if end is not None and t > end:
                    break
                pkt.pts -= base
                pkt.dts -= base
                pkt.stream = outs
                out.mux(pkt)
                last_t = t
                if progress and span > 0:
                    progress(min(1.0, max(0.0, (t - start) / span)))
        finally:
            out.close()
        if base is None:
            raise RuntimeError('자를 구간에 영상이 없습니다')
        offset = float(base * tb)
        return offset, max(0.0, last_t - offset)
    finally:
        inp.close()


def trim_in_place(path: Path, start: float, end: float | None, progress=None) -> tuple[float, float]:
    """path 파일 자체를 잘라 낸 결과로 바꾼다 (임시 파일에 만든 뒤 교체)"""
    path = Path(path)
    tmp = path.with_name(path.stem + '.trim' + path.suffix)
    try:
        res = trim_copy(path, tmp, start, end, progress)
        os.replace(tmp, path)
        return res
    finally:
        if tmp.exists():
            tmp.unlink()
