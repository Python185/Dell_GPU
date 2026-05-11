@echo off
REM UR_difference_check.py をタスクスケジューラから起動する用
REM このファイルは UR_difference_check.py と同じフォルダに置いてください。

chcp 65001 >nul
cd /d "%~dp0"

REM --- Python の指定（タスクでは PATH に Python が無いことが多いです）---
REM うまく動かない場合、次の 1 行の先頭 REM を外し、実際の python.exe のパスに書き換えてください。
set PYTHON_EXE=C:\Users\maki_\anaconda3\envs\maki2\python.exe

if defined PYTHON_EXE (
  "%PYTHON_EXE%" "G:\マイドライブ\MyWork\UR_difference_check.py"
  exit /b %ERRORLEVEL%
)

where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py -3 "%~dp0UR_difference_check.py"
  exit /b %ERRORLEVEL%
)

where python >nul 2>&1
if %ERRORLEVEL%==0 (
  python "%~dp0UR_difference_check.py"
  exit /b %ERRORLEVEL%
)
exit /b 1
