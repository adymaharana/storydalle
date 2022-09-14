if [ "$1" = "pororo" ]; then
  echo "Training on Pororo"
  DATA_DIR=../data/pororo/
  OUTPUT_ROOT=./out/pororo
  SENT_EMBED=512
  STORY_LEN=4
  LR=1e-4
  TRAIN_BS=1
  GRAD_ACC=4
  DATA_DIR=../data/flintstones
  OUTPUT_ROOT=./out/flintstones
  SENT_EMBED=512
  STORY_LEN=4
  LR=1e-5
  TRAIN_BS=1
  GRAD_ACC=4
elif [ "$1" = "didemo" ]; then
  echo "Training on DiDeMo"
  DATA_DIR=../data/didemo
  OUTPUT_ROOT=./out/didemo
  SENT_EMBED=512
  STORY_LEN=2
  TRAIN_BS=1
  GRAD_ACC=8
fi

#--prefix_model_name_or_path './1.3B/' \
#--model_name_or_path './1.3B/' \

python ./train_t2i.py \
--model_name_or_path './pretrained/' \
--tuning_mode 'story' \
--dataset_name $1 \
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
--learning_rate $LR \
--logging_steps 50 \
--eval_steps 500 \
--generate_steps 1000 \
--is_mega