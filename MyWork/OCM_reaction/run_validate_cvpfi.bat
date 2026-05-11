@echo off
echo Starting CVPFI validation...
cd /d "G:\マイドライブ\MyWork"
powershell -ExecutionPolicy Bypass -Command "& 'C:\Users\maki_\anaconda3\envs\maki2\python.exe' validate_cvpfi_cpu_gpu.py"
pause

