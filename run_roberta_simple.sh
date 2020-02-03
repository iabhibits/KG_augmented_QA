#! /bin/bash

python run_simple_roberta.py --task_name='arc' --data_dir="data/arc/processed" --model_type="Roberta" --model_name_or_path="roberta-large" --output_dir="save_roberta_simple/" --do_train --do_eval --do_test --per_gpu_train_batch_size=3 --gradient_accumulation_steps=1 --learning_rate=1e-5 --weight_decay=0.1 --adam_epsilon=1e-6 --max_grad_norm=1.0 --num_train_epochs=5.0 --logging_steps=1118 --save_steps=1118 --overwrite_output_dir  --seed=42 --max_ent_pre=262 --max_ent_hyp=83 --path_to_kg="./data/conceptnet" --max_seq_length=384 --cuda_device=3 --evaluate_during_training --warmup_steps=67 --overwrite_output_dir
#--overwrite_cache=False --no_cuda=True 
