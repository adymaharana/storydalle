if [ "$1" = "pororo" ]; then
  echo "Evaluating on Pororo"
  DATA_DIR=../data/pororo
  OUTPUT_ROOT=./out/pororo
  MODEL_CKPT=''
  SENT_EMBED=512
  STORY_LEN=4
elif [ "$1" = "flintstones" ]; then
  echo "Evaluating on Flintstones"
  DATA_DIR=../data/flintstones
  OUTPUT_ROOT=./out/flintstones
  MODEL_CKPT=''
  SENT_EMBED=512
  STORY_LEN=4
elif [ "$1" = "didemo" ]; then
  echo "Evaluating on DiDeMo"
  DATA_DIR=../data/didemo
  OUTPUT_ROOT=./out/didemo
  MODEL_CKPT=''
  SENT_EMBED=512
  STORY_LEN=2
fi


python ./infer_t2i.py \
--model_name_or_path  $MODEL_CKPT \
--prefix_model_name_or_path './1.3B/' \
--dataset_name $1 \
--tuning_mode story \
--dataset_name $1 \
--preseqlen 32 \
--condition \
--story_len $STORY_LEN \
--sent_embed $SENT_EMBED \
--prefix_dropout 0.2 \
--data_dir $DATA_DIR \
--dataloader_num_workers 1 \
--do_eval \
--per_gpu_eval_batch_size 16 \
--output_dir $OUTPUT_ROOT \
--mode $2
