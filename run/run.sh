# Preprocess
python [path_to_preprocess.py] \
--root [path_to_raw_data] \
--dest [path_to_input]

# Pretrain
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python [path_to_train.py] \
--batch_size 14 \
--data_path [path_to_input] \
--student_number [student_number] \
--distributed_world_size 4 \
--save_dir [path_to_output] \
--max_epoch 150 \
--wandb_entity [wandb_entity] \
--wandb_project [wandb_project] \
--valid_percent 0.1 \
--pretrain

# Finetune
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python [path_to_train.py] \
--batch_size 14 \
--data_path [path_to_input] \
--student_number [student_number] \
--distributed_world_size 4 \
--save_dir [path_to_output] \
--max_epoch 150 \
--wandb_entity [wandb_entity] \
--wandb_project [wandb_project] \
--valid_percent 0.1 \
--finetune \
--load_dir [pretrain_load_dir] \
--load_ckpt [pretrain_ckpt]
