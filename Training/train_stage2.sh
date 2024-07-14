EXP_NAME="train_stage2"

accelerate launch train_stage2.py \
 --pretrained_model_name_or_path="./ckpts/stable-video-diffusion-img2vid-xt-1-1" \
 --controlnet_model_name_or_path="{YOUR_STAGE1_MODEL_PATH}/controlnet" \
 --output_dir="./logs/${EXP_NAME}/" \
 --height=384 \
 --width=384 \
 --train_height=320 \
 --train_width=320 \
 --seed=42 \
 --learning_rate=2e-5 \
 --per_gpu_batch_size=1 \
 --num_train_epochs=5 \
 --max_train_steps=1000000 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=1 \
 --checkpointing_steps=2500 \
 --checkpoints_total_limit=100 \
 --validation_steps=2500 \
 --gradient_checkpointing \
 --num_validation_images=4 \
 --use_8bit_adam \
 --sample_stride=4 \
 --num_frames=25 \