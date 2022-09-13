#python eval_char_clf.py --dataset pororo --img_ref_dir /nas-ssd/adyasha/datasets/pororo_png/ --img_gen_dir /nas-ssd/adyasha/out/minDALLEs/pororo/test_images/images/ --model_name inception --model_path ./out/pororo-epoch-10.pt --mode test --num_classes 9

python eval_char_clf.py --dataset flintstones --img_ref_dir /nas-ssd/adyasha/datasets/flintstones/ --img_gen_dir /nas-ssd/adyasha/out/minDALLEs/flintstones/test_images/images/ --model_name inception --model_path /playpen-ssd/adyasha/projects/StoryGAN/classifier/models/inception_32_1e-05/best.pt --mode test --num_classes 7
