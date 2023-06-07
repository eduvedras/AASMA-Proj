#!/bin/bash

# Install requirements
pip install -r requirements.txt

if [ "$1" = "qlearning" ]; then
    python train.py
elif [ "$1" = "sarsa" ]; then
    python train.py agent=sarsa
else
    echo "Invalid flag provided."
fi