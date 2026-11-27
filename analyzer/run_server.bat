@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title 배구 영상 인식 서버

rem ---- 처음 한 번: 파이썬 가상환경과 라이브러리 설치 (몇 분, 약 5GB) ----
if not exist ".venv\Scripts\python.exe" (
  echo [설치] 파이썬 가상환경을 만듭니다...
  py -3.12 -m venv .venv 2>nul || py -3 -m venv .venv 2>nul || python -m venv .venv
  if not exist ".venv\Scripts\python.exe" (
    echo 파이썬을 찾을 수 없습니다. https://www.python.org 에서 Python 3.12 를 설치한 뒤 다시 실행하세요.
    pause
    exit /b 1
  )
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  where nvidia-smi >nul 2>nul
  if not errorlevel 1 (
    echo [설치] NVIDIA GPU 용 PyTorch 를 설치합니다...
    ".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  )
  echo [설치] 영상 인식 라이브러리를 설치합니다...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo 설치에 실패했습니다. 인터넷 연결을 확인하고 .venv 폴더를 지운 뒤 다시 실행하세요.
    pause
    exit /b 1
  )
)

rem ---- 켤 때마다: 새로 추가된 라이브러리(예: 영상 자르기용 PyAV)가 빠져 있으면 설치 ----
".venv\Scripts\python.exe" -c "import av, cv2, ultralytics, yt_dlp, pandas, openpyxl, lap, numpy" >nul 2>nul
if errorlevel 1 (
  echo [설치] 빠진 라이브러리를 설치합니다...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo 라이브러리 설치에 실패했습니다. 인터넷 연결을 확인한 뒤 다시 실행하세요.
    pause
    exit /b 1
  )
)

rem ---- 서버 실행 (이 창을 닫으면 꺼짐) ----
".venv\Scripts\python.exe" -u server.py %*
pause
