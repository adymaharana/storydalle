python ./finetune_mscoco.py \
--model_name_or_path './1.3B/' \
--data_dir ./mscoco2014 \
--dataloader_num_workers 16 \
--output_dir ./out/minDALLE_mscoco \
--do_train --do_eval \
--per_gpu_train_batch_size 2 \
--per_gpu_eval_batch_size 2 \
--overwrite_output_dir \
--num_train_epochs 5 \
--gradient_accumulation_steps 4