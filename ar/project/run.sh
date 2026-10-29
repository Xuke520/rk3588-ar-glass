#!/bin/bash
eval "$(/home/elf/miniconda3/bin/conda shell.bash hook)"
conda activate ar
python /home/elf/ar/python/main.py

./home/elf/ar/build/main
