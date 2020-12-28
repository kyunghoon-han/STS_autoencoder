#!/usr/bash

CUDA_VISIBLE_DEVICES=3 python3 runner.py --batch_size 64 --epochs 1000 --num_test 100 --lr_stt 0.01 --lr_tts 0.0001 --step_lrtts 20 --step_lrstt 10 --STS_threshold 10 --verbose 5 --data_dir ./preprocessing_unit/preprocessed --encoded_txt_pkl encoded_hangul.pkl --log_dir ./logs --log_filename logs.csv --model_save_dir ./models --device cuda --device_auto True
