#! /bin/bash
loss="diceFIMce"
alpha=1.0
l_fim=0.1

python main.py \
    --config 'config.prostate_mri.fedclam' \
    --loss "$loss" \
    --l_fim "$l_fim" \
    --fim_warmup 100 \
    --alpha "$alpha" \
    --beta 1 \
    --run_notes  "final" \
    --exp_name "$_${loss}_alpha${alpha}_lfim${l_fim}"