export LR=0.00001
export WARMUP_UPDATES=150
export MAX_UPDATES=8000
export BSZ=16
export SEED=42
CUDA_VISIBLE_DEVICES=0 fairseq-train --ddp-backend=legacy_ddp \
    ../data/social_iqa/ \
    --user-dir ./examples/roberta/social_iqa --restore-file ../models/roberta.base/model.pt \
    --reset-optimizer --reset-dataloader --reset-meters --no-epoch-checkpoints --no-last-checkpoints \
    --no-save-optimizer-state --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --task social_iqa --init-token 0 --bpe gpt2 --arch roberta_base --max-positions 512 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --criterion sentence_ranking \
    --num-classes 3 --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --warmup-updates $WARMUP_UPDATES --total-num-update $MAX_UPDATES \
    --batch-size $BSZ --max-update $MAX_UPDATES --log-format simple --log-interval 25 --seed $SEED \
    --save-dir ../outputs/social_iqa/roberta_base_prefix_10_12/output_${LR}_${BSZ}_${MAX_UPDATES}_${SEED} --knowledge_layer 9 12