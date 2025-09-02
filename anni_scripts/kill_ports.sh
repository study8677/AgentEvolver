#!/bin/bash

PORTS=(8000 8001)

for PORT in "${PORTS[@]}"
do
    # 使用 netstat 查找监听该端口的进程 PID
    PID=$(netstat -tulnp 2>/dev/null | grep ":$PORT" | awk '{print $7}' | cut -d'/' -f1)

    if [ ! -z "$PID" ]; then
        echo "Killing process on port $PORT (PID: $PID)"
        kill -9 $PID
    else
        echo "No process found on port $PORT"
    fi
done
