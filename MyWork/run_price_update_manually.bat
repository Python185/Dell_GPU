@echo off
echo Starting price update...
cd /d "G:\マイドライブ\MyWork"
echo [%date% %time%] Starting price update (PowerShell version)... >> price_update_batch.log
powershell -ExecutionPolicy Bypass -Command "& 'C:\Users\maki_\anaconda3\envs\maki2\python.exe' My_rule.py --auto"
echo [%date% %time%] PowerShell execution completed >> price_update_batch.log
pause 