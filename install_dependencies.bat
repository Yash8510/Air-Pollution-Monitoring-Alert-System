@echo off
echo ============================================================
echo  Installing Air Quality Monitor Server Dependencies
echo ============================================================
echo.

cd server

echo Installing Flask and required packages...
python -m pip install Flask==3.0.0 Werkzeug==3.0.1
echo.

echo Installing AI/ML packages...
python -m pip install numpy==1.26.4
echo.

python -m pip install scikit-learn==1.4.0
echo.

echo ============================================================
echo  Installation complete!
echo ============================================================
echo.
echo To start the server, run:
echo    cd server
echo    python dashboard_server.py
echo.
pause
