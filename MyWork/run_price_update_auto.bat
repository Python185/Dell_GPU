@echo off
chcp 65001 >nul
cd /d "G:\マイドライブ\MyWork"
echo [%date% %time%] Starting price update with correct Python path... >> price_update_batch.log

REM 正しいAnaconda環境のPythonを使用
if exist "C:\Users\maki_\anaconda3\envs\maki2\python.exe" (
    echo [%date% %time%] Using Python from Anaconda maki2 environment >> price_update_batch.log
    "C:\Users\maki_\anaconda3\envs\maki2\python.exe" My_rule.py --auto >> price_update_batch.log 2>&1
) else (
    echo [%date% %time%] Anaconda Python not found, trying PowerShell >> price_update_batch.log
    powershell -Command "python My_rule.py --auto" >> price_update_batch.log 2>&1
)

if %errorlevel% equ 0 (
    echo [%date% %time%] Price update completed successfully >> price_update_batch.log
) else (
    echo [%date% %time%] Price update failed with error code %errorlevel% >> price_update_batch.log
)
exit /b %errorlevel% 