#!/bin/bash

for ((i=0; i<5; i++))
do
    echo $(python3 MM_for.py)
done

for ((i=0; i<5; i++))
do
    echo $(pypy3 MM_for.py)
done

for ((i=0; i<5; i++))
do
    echo $(pypy3 MM_numpy.py)
done