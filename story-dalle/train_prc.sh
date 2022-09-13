if [ "$1" = "mscoco" ]; then
  echo "Training on MSCOCO"
  DATA_DIR=/nas-ssd/adyasha/datasets/mscoco2014
  OUTPUT_ROOT=/nas-ssd/adyasha/models/minDALLEprc/mscoco2014
elif [ "$1" = "pororo" ]; then
  echo "Training on Pororo"
  DATA_DIR=/nas-ssd/adyasha/datasets/pororo_png
  OUTPUT_ROOT=/nas-ssd/adyasha/models/minDALLEprc/pororo
elif [ "$1" = "flintstones" ]; then
  echo "Training on Flintstones"
  DATA_DIR=/nas-ssd/adyasha/datasets/flintstones
  OUTPUT_ROOT=/nas-ssd/adyasha/models/minDALLEprc/flintstones
elif [ "$1" = "mpii" ]; then
  echo "Training on MPII"
  DATA_DIR=/nas-ssd/adyasha/datasets/mpii
  OUTPUT_ROOT=/nas-ssd/adyasha/models/minDALLEprc/mpii
elif [ "$1" = "didemo" ]; then
  echo "Training on DiDeMo"
  DATA_DIR=/nas-ssd/adyasha/datasets/didemo
  OUTPUT_ROOT=/nas-ssd/adyasha/models/minDALLEprc/didemo
fi

#--prefix_model_name_or_path './1.3B/' \
#--model_name_or_path './1.3B/' \

python ./train_t2i.py \
--prefix_model_name_or_path './1.3B/' \
--tuning_mode story \
--dataset_name $1 \
--preseqlen 32 \
--prefix_dropout 0.2 \
--data_dir $DATA_DIR \
--dataloader_num_workers 4 \
--output_dir $OUTPUT_ROOT \
--log_dir /nas-ssd/adyasha/runs/ \
--do_train --do_eval \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 4 \
--overwrite_output_dir \
--num_train_epochs 100 \
--gradient_accumulation_steps 4 \
--learning_rate 0.1
