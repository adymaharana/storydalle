python ./train_prefix.py \
--prefix_model_name_or_path './1.3B/' \
--preseqlen 10 \
--prefix_dropout 0.2 \
--data_dir ../StoryGAN/pororo_png \
--dataloader_num_workers 2 \
--output_dir ./out/minDALLE_prefix_pororo \
--do_train --do_eval \
--per_gpu_train_batch_size 2 \
--per_gpu_eval_batch_size 2