@echo off
echo ====================================================
echo Switching to project directory...
cd /d "E:\Projects\Projects\Multi-Agent"

echo ====================================================
echo [1/3] Starting FastAPI Coordinator Server...
start "FastAPI Server" cmd /k "py -3.12 -m fastapi dev backend/coordinator/main.py"

echo ====================================================
echo [2/3] Starting Python Handler Service...
start "Handler Service" cmd /k "py -3.12 -m backend.handler.handler_service"

echo ====================================================
echo All processes launched successfully!
echo To close everything, close the 2 server windows and exit the game.
exit