DATA_DIR="/workspace/data/videodata"

VIDEO_NAME="coral-reef-long"
OBJECT="person"


python train_difference_model.py \
  --csv_in $DATA_DIR/csv/${VIDEO_NAME}.csv \
  --csv_out_base $DATA_DIR/cnn_models/${VIDEO_NAME}-out \
  --video_in $DATA_DIR/videos/${VIDEO_NAME}.mp4 \
  --frame_delay 15 \
  --num_frames 1000 \
  --object $OBJECT \
  --scale 0.1 \
  --features hog