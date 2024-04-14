#!/bin/bash

for script in /data/vitsGPT/vits/scripts/*.sh; do
    bash "$script" || true
done
