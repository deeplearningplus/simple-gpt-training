export CUDA_VISIBLE_DEVICES=4,5,6,7
    ##--fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'OPTDecoderLayer' \
torchrun --nproc_per_node=4 --master_port=8080 train.py \
    --data_path data/text_reports.txt \
    --bf16 True \
    --output_dir checkpoint \
    --num_train_epochs 30 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 24 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 3e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --adam_beta1 0.90 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to tensorboard \
    --tf32 True \
    --dataloader_num_workers 2 \
    --dataloader_persistent_workers True

