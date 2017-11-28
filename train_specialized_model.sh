DATA_DIR="/workspace/data/videodata"

VIDEO_NAME="coral-reef-long"
OBJECT="person"
NUM_FRAMES="1188000"
START_FRAME="648000"
GPU_NUM="0"


python train_specialized_model.py \
  --avg_fname ${VIDEO_NAME}.npy \
  --csv_in $DATA_DIR/csv/${VIDEO_NAME}.csv \
  --video_in $DATA_DIR/videos/${VIDEO_NAME}.mp4 \
  --output_dir $DATA_DIR/cnn-models/ \
  --base_name ${VIDEO_NAME} \
  --objects $OBJECT \
  --num_frames $NUM_FRAMES \
  --start_frame $START_FRAME