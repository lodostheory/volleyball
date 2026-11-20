"""배구공 전용 탐지 가중치 만들기 (추가 학습 · fine-tuning)

일반 COCO 모델의 'sports ball' 은 멀리 있는 작은 배구공을 자주 놓칩니다. 배구공 사진에 상자를 그린
YOLO 형식 데이터셋으로 추가 학습하면 크게 좋아집니다. 학습이 끝나면 가중치를 models/volleyball_ball.pt 로
복사하므로, 분석 서버가 다음부터 자동으로 이 가중치를 씁니다.

데이터셋 구하는 곳 (예)
  - AI-Hub '스포츠 영상 데이터(배구)' 등 라벨링된 데이터를 신청해 YOLO 형식으로 변환
  - Roboflow Universe 의 volleyball 데이터셋 (YOLOv8 형식으로 내려받기)
  - 직접 찍은 영상에서 프레임을 뽑아 라벨링 도구(CVAT, Label Studio 등)로 공 상자를 그리기

data.yaml 예:
  path: D:/datasets/volleyball
  train: images/train
  val: images/val
  names: {0: volleyball}

실행:
  .venv\\Scripts\\python train_ball.py --data D:/datasets/volleyball/data.yaml --epochs 80
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description='배구공 탐지 가중치 추가 학습')
    ap.add_argument('--data', required=True, help='YOLO 형식 data.yaml 경로')
    ap.add_argument('--base', default='yolo11s.pt', help='시작 가중치 (yolo11n/s/m.pt)')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--imgsz', type=int, default=1280, help='작은 공을 위해 큰 입력 크기 권장')
    ap.add_argument('--batch', type=int, default=4, help='GPU 메모리 4GB 기준. 메모리가 부족하면 2 로')
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    base = Path(a.base)
    if not base.exists():
        (HERE / 'models').mkdir(exist_ok=True)
        base = HERE / 'models' / a.base
    model = YOLO(str(base))
    res = model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                      project=str(HERE / 'data' / 'train'), name='ball', exist_ok=True)
    best = Path(res.save_dir) / 'weights' / 'best.pt'
    dst = HERE / 'models' / 'volleyball_ball.pt'
    shutil.copy(best, dst)
    print(f'완료: {dst} — 분석 서버를 다시 켜면 이 가중치를 씁니다')


if __name__ == '__main__':
    main()
