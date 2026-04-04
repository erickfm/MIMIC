#!/bin/bash
cd /home/erick/projects/FRAME
python3 tools/run_hal_model.py \
  --checkpoint checkpoints/hal_original.pt \
  --dolphin-path "/home/erick/projects/hal/emulator/squashfs-root/usr/bin/dolphin-emu" \
  --iso-path "/home/erick/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).iso" \
  --character FOX \
  --cpu-character FOX
