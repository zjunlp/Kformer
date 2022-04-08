export LR=0.00001
export WARMUP_UPDATES=150
export MAX_UPDATES=12000
export BSZ=8
export SEED=33
CUDA_VISIBLE_DEVICES=3 fairseq-train --ddp-backend=legacy_ddp \
    ../data/tw_med/ \
    --user-dir ./examples/roberta/med_qa --restore-file ../models/roberta.base/model.pt \
    --reset-optimizer --reset-dataloader --reset-meters --no-epoch-checkpoints --no-last-checkpoints \
    --no-save-optimizer-state --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --task med_qa --init-token 0 --bpe gpt2 --arch roberta_base --max-positions 512 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --criterion sentence_ranking \
    --num-classes 4 --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --warmup-updates $WARMUP_UPDATES --total-num-update $MAX_UPDATES \
    --batch-size $BSZ --max-update $MAX_UPDATES --log-format simple --log-interval 25 --seed $SEED \
    --save-dir ../outputs/med_qa/roberta_base_prefix_9_12/output_${LR}_${BSZ}_${MAX_UPDATES}_${SEED}  --knowledge_layer 9 12

CUDA_VISIBLE_DEVICES=3 python test_med.py ../outputs/med_qa/roberta_base_prefix_9_12/output_${LR}_${BSZ}_${MAX_UPDATES}_${SEED}  --knowledge_layer 9 12