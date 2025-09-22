@echo off
echo Starting server...
start "Server" cmd /k python server_1.py

echo Waiting for server to start...
timeout /t 3 /nobreak

for /L %%i in (0,1,4) do (
    echo Starting client %%i
    start "Client %%i" cmd /k python client_1.py %%i
)

echo All clients have been launched.