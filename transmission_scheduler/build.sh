#!/bin/bash
OS=$(uname -s)
if [ "$OS" == "Linux" ]; then
    echo "Building for Linux (libtransmission_scheduler.so)..."
    gcc -O3 -fPIC -shared transmission_scheduler.c -o libtransmission_scheduler.so
elif [ "$OS" == "Darwin" ]; then
    echo "Building for macOS (libtransmission_scheduler.dylib)..."
    gcc -O3 -fPIC -shared transmission_scheduler.c -o libtransmission_scheduler.dylib
else
    echo "Building generic (libtransmission_scheduler.so)..."
    gcc -O3 -fPIC -shared transmission_scheduler.c -o libtransmission_scheduler.so
fi