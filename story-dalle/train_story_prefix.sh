if [ "$1" = "pororo" ]; then
  echo "Training on Pororo"
  DATA_DIR=/nas-ssd/adyasha/datasets/pororo_png
  OUTPUT_ROOT=/nas-ssd/adyasha/modelsp/minDALLEsp/pororo
  SENT_EMBED=128
  TRAIN_BS=2
  STORY_LEN=4
  GRAD_ACC=4
elif [ "$1" = "flintstones" ]; then
  echo "Training on Flintstones"
  DATA_DIR=/nas-ssd/adyasha/datasets/flintstones
  OUTPUT_ROOT=/nas-ssd/adyasha/models/minDALLEsp/flintstones
  SENT_EMBED=512
  STORY_LEN=4
  LR=1e-5
elif [ "$1" = "mpii" ]; then
  echo "Training on MPII"
  DATA_DIR=/nas-ssd/adyasha/datasets/mpii
  OUTPUT_ROOT=/nas-ssd/adyasha/models/minDALLEsp/mpii
  SENT_EMBED=128
  STORY_LEN=4
elif [ "$1" = "didemo" ]; then
  echo "Training on DiDeMo"
  DATA_DIR=/nas-ssd/adyasha/datasets/didemo
  OUTPUT_ROOT=/nas-ssd/adyasha/models/minDALLEsp/didemo
  SENT_EMBED=512
  STORY_LEN=2
  TRAIN_BS=1
  GRAD_ACC=8
fi

#--prefix_model_name_or_path './1.3B/' \
#--model_name_or_path './1.3B/' \

python ./train_t2i.py \
--prefix_model_name_or_path './1.3B/' \
--tuning_mode story \
--prompt \
--dataset_name $1 \
--preseqlen 32 \
--condition \
--story_len $STORY_LEN \
--sent_embed $SENT_EMBED \
--prefix_dropout 0.2 \
--data_dir $DATA_DIR \
--dataloader_num_workers 4 \
--output_dir $OUTPUT_ROOT \
--log_dir /nas-ssd/adyasha/runs/ \
--do_train --do_eval \
--per_gpu_train_batch_size $TRAIN_BS \
--per_gpu_eval_batch_size 2 \
--num_train_epochs 50 \
--gradient_accumulation_steps $GRAD_ACC \
--learning_rate 1e-3 \
--logging_steps 50 \
--eval_steps 500 \
--generate_steps 1000
