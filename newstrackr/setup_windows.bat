@echo off
title Newstrackr Setup

echo.
echo ========================================
echo       Newstrackr Setup for Windows
echo ========================================
echo.

echo Installing dependencies...
python install_deps.py

echo.
echo ========================================
echo Setup complete! Starting the app...
echo ========================================
echo.

echo The app will open in your browser shortly.
echo If it doesn't open automatically, go to: http://localhost:8501
echo.

streamlit run app.py

pause