@echo off
cd /d "%~dp0"
title PCB Calibration Tool
color 0B
echo =====================================================================
echo       PCB CALIBRATION TOOL (OpenCV 4-Corners)
echo =====================================================================
echo.
echo [*] Running calibrate.py ...
echo.
python calibrate.py
pause
