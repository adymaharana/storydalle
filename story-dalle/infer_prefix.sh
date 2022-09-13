if [ "$1" = "mscoco" ]; then
  echo "Training on MSCOCO"
  DATA_DIR=/nas-ssd/adyasha/datasets/mscoco2014
  MODEL_CKPT=./out/minDALLE_prefix_mscoco/21022022_000705/ckpt/mscoco-prefix-epoch=04.ckpt
elif [ "$1" = "pororo" ]; then
  echo "Training on Pororo"
  DATA_DIR=/nas-ssd/adyasha/datasets/pororo_png
  MODEL_CKPT='/nas-ssd/adyasha/models/minDALLEp/pororo/Model/2.pth'
elif [ "$1" = "flintstones" ]; then
  echo "Training on Flintstones"
  DATA_DIR=/nas-ssd/adyasha/datasets/flintstones
  OUTPUT_ROOT=/nas-ssd/adyasha/models/minDALLEp/flintstones
elif [ "$1" = "mpii" ]; then
  echo "Training on MPII"
  DATA_DIR=/nas-ssd/adyasha/datasets/mpii
  OUTPUT_ROOT=/nas-ssd/adyasha/models/minDALLEp/mpii
elif [ "$1" = "didemo" ]; then
  echo "Training on DiDeMo"
  DATA_DIR=/nas-ssd/adyasha/datasets/didemo
  OUTPUT_ROOT=/nas-ssd/adyasha/models/minDALLEp/didemo
fi

python ./infer_prefix.py \
--model_name_or_path  $MODEL_CKPT \
--prefix_model_name_or_path './1.3B/' \
--dataset_name $1 \
--preseqlen 16 \
--prefix_dropout 0.2 \
--data_dir $DATA_DIR \
--dataloader_num_workers 16 \
--do_eval \
--per_gpu_eval_batch_size 2 \
