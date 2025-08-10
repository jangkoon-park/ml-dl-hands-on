@echo off
setlocal EnableExtensions
chcp 65001 >NUL

REM ===== 설정 =====
REM 로컬 레포 경로
set "ROOT=C:\projects\ml-dl-hands-on"
REM 원격 GitHub URL
set "GIT_URL=https://github.com/jangkoon-park/ml-dl-hands-on.git"
REM =================

cd /d "%ROOT%"

REM Git 설치 확인
where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git이 설치되어 있지 않습니다.
    pause
    exit /b 1
)

REM Git 초기화(최초 1회만)
if not exist ".git" (
    git init
)

REM 브랜치 설정
git branch -M main

REM 원격 주소 등록(기존 있으면 재설정)
git remote remove origin 2>nul
git remote add origin "%GIT_URL%"

REM 원격 최신 이력 가져오기
git fetch origin
git rebase --abort 2>NUL
git merge --abort 2>NUL
git pull --rebase --autostash origin main || git pull --allow-unrelated-histories origin main

REM 변경 사항 스테이징 및 커밋
git add -A
git commit -m "Add chapter folders and README files" 2>NUL

REM 원격에 푸시
git push -u origin main

echo.
echo [DONE] 원격 GitHub에 커밋 및 푸시 완료!
pause
