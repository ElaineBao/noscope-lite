DATA_DIR="/workspace/data/videodata"

VIDEO_NAME="coral-reef-long"
OBJECT="person"
NUM_FRAMES="1188000"
START_FRAME="648000"
GPU_NUM="0"


python train_difference_model.py \
  --csv_in $DATA_DIR/csv/${VIDEO_NAME}.csv \
  --csv_out_base $DATA_DIR/cnn_models/${VIDEO_NAME}_out.csv
  --video_in $DATA_DIR/videos/${VIDEO_NAME}.mp4 \
  --frame_delay 15 \
  --objects $OBJECT \
  --scale 0.1 \
  --features hog \