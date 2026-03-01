# python main.py \
#     --config 'config.fundus.fedavg' \
#     --run_notes 'baselines_lr_comp' \
#     --batch_size 8 \
#     --exp_name 'fedavg'

python main.py \
    --config 'config.fundus.fedevi' \
    --run_notes 'baselines_in' \
    --exp_name 'fedevi'

# python main.py \
#     --config 'config.fundus.fedavgm' \
#     --run_notes 'baselines_in' \
#     --exp_name 'fedavgm'

# python main.py \
#     --config 'config.fundus.fedavgm' \
#     --run_notes 'baselines_in_clam_ablation' \
#     --loss 'diceFIMfundus' \
#     --exp_name 'fedavgm'

# python main.py \
#     --config 'config.fundus.fedfa' \
#     --run_notes 'baselines_in' \
#     --exp_name 'fedfa'

# python main.py \
#     --config 'config.fundus.fedsam' \
#     --run_notes 'baselines_in' \
#     --exp_name 'fedsam'

# python main.py \
#     --config 'config.fundus.fedharmo' \
#     --run_notes 'baselines_in' \
#     --exp_name 'fedharmo' &
# python main.py \
#     --config 'config.fundus.feddyn' \
#     --run_notes 'baselines_in' \
#     --exp_name 'feddyn'



# python main.py \
#     --config 'config.fundus.fedclam' \
#     --loss 'diceFIMfundus' \
#     --l_fim 0.01 \
#     --alpha 1 \
#     --beta 1 \
#     --exp_name "fedclam_fim" \
#     --run_notes 'baselines_in_sota' \

# python main.py \
#     --config 'config.fundus.fedclam' \
#     --loss 'diceFIMfundus' \
#     --l_fim 0.00 \
#     --alpha 1 \
#     --beta 1 \
#     --exp_name "fedclam_fim" \
#     --run_notes 'baselines_in_no_fim_ablation' \