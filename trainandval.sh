#!/bin/bash

if [ $# -eq 0 ]; then
    python src/python/trainandval.py
elif [ $# -eq 1 ]; then
    python src/python/trainandval.py $1
fi