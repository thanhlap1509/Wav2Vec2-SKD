pretrained_model = 'nguyenvulebinh/wav2vec2-base-vietnamese-250h'
# distill
distill_layer = 8

# path to weight file for inference
eval_model = 'path_to_wav2vec2_model_pth_file'

# dataset
train_data = 'path_to_train_data_csv_file'
eval_data = 'path_to_eval_data_csv_file'

# ctc
save_dir = 'save_directory'

# training args
num_epochs = 30
learning_rate = 3e-5
per_device_train_batch_size = 2
gradient_accumulation_steps = 2
warmup_ratio = 0.1
# spec augment
apply_spec_augment = True
mask_time_prob = 0.05
mask_feature_prob = 0.0

# ctc config
ctc_loss_reduction = 'mean'
