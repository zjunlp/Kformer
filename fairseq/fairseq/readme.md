## Knowledge Infusion

1. fairseq\modules\transformer_sentence_encoder_layer.py

   We add two Linear Layer for Knowledge in this code and do the knowledge infusion in the forward function. 

2.   fairseq\modules\transformer_sentence_encoder.py

   In the encoder part, we add the knowledge_embedding part, and do a simple retrieval in this part.(line 253)

3.  fairseq\models\fairseq_model.py

   To initialize the knowledge embedding with the roberta embedding layer , we add the parameter in the state_dict. (line 115-119) . 

4. We add an arg knowledge_layer to choose which layer we do knowledge infusion.

   For example, --knowledge layer 10 11 means we add knowledge in layer [10,11)

5. examples\roberta\commonsense_qa

   Modify the model's input. It can be changed as what you need.



```bash
- fairseq-train --fp16 --ddp-backend=legacy_ddp 
$Data_path$
--user-dir ./examples/roberta/commonsense_qa --restore-file $Model_path$
--reset-optimizer --reset-dataloader --reset-meters --no-epoch-checkpoints --no-last-checkpoints
--no-save-optimizer-state --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric 
--task commonsens_qa --init-token 0 --bpe gpt2 --arch roberta_large --max-positions 512 
--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --criterion sentence_ranking 
--num-classes 5 --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 
--lr-scheduler polynomial_decay --lr $LR$ --warmup-updates $WARMUP_UPDATES$ --total-num-update $MAX_UPDATES$
--batch-size $BSZ$ --max-update $MAX_UPDATES$ --log-format simple --log-interval 25 --seed $SEED$
--save-dir $OUTPUT_DIR$ --knowledge_layer 23 24

```