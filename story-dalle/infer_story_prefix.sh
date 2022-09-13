if [ "$1" = "pororo" ]; then
  echo "Training on Pororo"
  DATA_DIR=/nas-ssd/adyasha/datasets/pororo_png
  OUTPUT_ROOT=/nas-ssd/adyasha/out/minDALLEpr/pororo/
  MODEL_CKPT='/nas-ssd/adyasha/models/minDALLEs/pororo/Model/0.pth'
  SENT_EMBED=128
  STORY_LEN=4
elif [ "$1" = "flintstones" ]; then
  echo "Training on Flintstones"
  DATA_DIR=/nas-ssd/adyasha/datasets/flintstones
  OUTPUT_ROOT=/nas-ssd/adyasha/out/minDALLEpr/flintstones
  MODEL_CKPT='/nas-ssd/adyasha/models/minDALLEs/flintstones/Model/1.pth'
  SENT_EMBED=512
  STORY_LEN=4
elif [ "$1" = "mpii" ]; then
  echo "Training on MPII"
  DATA_DIR=/nas-ssd/adyasha/datasets/mpii
  OUTPUT_ROOT=/nas-ssd/adyasha/models/minDALLEpr/mpii
  SENT_EMBED=128
  STORY_LEN=4
elif [ "$1" = "didemo" ]; then
  echo "Training on DiDeMo"
  DATA_DIR=/nas-ssd/adyasha/datasets/didemo
  OUTPUT_ROOT=/nas-ssd/adyasha/out/minDALLEpr/didemo
  MODEL_CKPT='/nas-ssd/adyasha/models/minDALLEs/didemo/Model/0.pth'
  SENT_EMBED=512
  STORY_LEN=2
fi

python ./infer_prefix.py \
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
--mode 'test'
