python inference_opendomain.py \
--audio_path="demo/audios/000001.wav" \
--img_path="demo/images/000001.jpg" \
--ckpt_dir="ckpts/mofa/ldmk_controlnet" \
--save_root="results" \
--max_frame_len=125 \
--ldmk_render="aniportrait"