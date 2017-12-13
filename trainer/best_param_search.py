#!/usr/bin/env python

from utils import accuracy as acc
from utils import optimizer as opt
import datetime
import stat
import json
import os

# Dir Structure (e.g. video is coral-reef-long.mp4)
# DATA_DIR_PREFIX |-- EXPERIMENTS_DIR_PREFIX (experiments/) |-- experiment_dir (coral-reef-long/) |- TRAIN_DIRNAME (train/) |- pipeline_path (coral-reef-long_convnet_32_32_1.pb-non_blocked_mse.src/) |- val_csv_path (val_1836000_3024000.csv)
#                                                                                                                                                                                                      |- val_log_path (val_1836000_3024000.log)
#                                                                                                                                                                                                      |- run_optimizer_script (run_optimizerset.sh)
#                                                                                                 |- summary_file (summary.csv)
#                                                                                                 |- test_path (error_rate e.g. 0.25/,0.1/,...) |- test_csv_filename (test_3024000_4212000.csv)
#                                                                                                                                               |- test_log_filename (test_3024000_4212000.log)
#                                                                                                                                               |- best_params_file (params.json)
#                                                                                                                                               |- run_test_script (run_testset.sh)
#                 |-- VIDEO_DIR_PREFIX (videos/) |-- video_path (coral-reef-long.mp4)
#                 |-- TRUTH_DIR_PREFIX (csv/) |-- truth_csv (coral-reef-long.csv)
#                 |-- DD_MEAN_DIR_PREFIX (dd-means/)
#                 |-- CNN_MODEL_DIR_PREFIX (cnn-models/) |-- cnn_path (coral-reef-long_convnet_32_32_1.pb)
#                                                        |-- cnn_path (coral-reef-long_convnet_32_32_2.pb)
#                                                        |-- large_cnn_path (faster_rcnn_resnet101_coco_model.pb)
#                 |-- CNN_AVG_DIR_PREFIX (cnn-avg/) |-- cnn_avg_path (coral-reef-long.txt)

DATA_DIR_PREFIX = '/workspace/data/videodata'
EXPERIMENTS_DIR_PREFIX = os.path.join(DATA_DIR_PREFIX, 'experiments/')
VIDEO_DIR_PREFIX = os.path.join(DATA_DIR_PREFIX, 'videos')
TRUTH_DIR_PREFIX = os.path.join(DATA_DIR_PREFIX, 'csv')
DD_MEAN_DIR_PREFIX = os.path.join(DATA_DIR_PREFIX, 'dd-means')
CNN_MODEL_DIR_PREFIX = os.path.join(DATA_DIR_PREFIX, 'cnn-models')
CNN_AVG_DIR_PREFIX = os.path.join(DATA_DIR_PREFIX, 'cnn-avg')
large_cnn_path = os.path.join(CNN_MODEL_DIR_PREFIX,'faster_rcnn_resnet101_coco_model.pb')

TRAIN_DIRNAME = 'train'
RUN_TEST_SCRIPT_FILENAME = 'run_testset.sh'
RUN_OPTIMIZER_SCRIPT_FILENAME = 'run_optimizerset.sh'
OUTPUT_SUMMARY_CSV = 'summary.csv'

NOSCOPE_APP_PATH = '/workspace/data/tensorflow/tensorflow/noscope-lite/noscope'
GPU_NUM = 0

VIDEO_INFO = dict()
VIDEO_INFO['video_name'] = "coral-reef-long"
VIDEO_INFO['video_postfix'] = '.mp4'
VIDEO_INFO['object_id'] = 1
VIDEO_INFO['object_name'] = 'person'
VIDEO_INFO['pipelines'] = [("coral-reef-long_convnet_32_32_1.pb", None),
	                        ("coral-reef-long_convnet_32_32_2.pb", None), ]
VIDEO_INFO['val_start'] = 648000 + 1188000
VIDEO_INFO['val_len'] = 1188000
VIDEO_INFO['test_start'] = 648000 + 1188000 * 2
VIDEO_INFO['test_len'] = 1188000

NO_CACHING = False
TARGET_ERROR_RATES = [0.25, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]

RUN_BASH_SCRIPT = """
#!/bin/bash

# {date}

if [[ -z $1 ]]; then
    echo "Usage: $0 GPU_ID"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$1"

time {noscope_app_path} \\
    --diff_thresh={dd_thres} \\
    --small_cnn_thresh_lower={small_cnn_lower_thres} \\
    --small_cnn_thresh_upper={small_cnn_upper_thres} \\
    --large_cnn_thresh={large_cnn_thres} \\
    --target_object_id={object_id}    \\
    --skip_small_cnn={skip_small_cnn} \\
    --skip_diff_detection={skip_dd} \\
    --skip={kskip} \\
    --avg_fname={cnn_avg_path} \\
    --small_cnn_graph={small_cnn_path} \\
    --large_cnn_graph={large_cnn_path}  \\
    --video={video_path} \\
    --confidence_csv={output_csv} \\
    --start_from={start_frame} \\
    --nb_frames={nb_frames} \\
    --diff_detection_weights={diff_detection_weights} \\
    --use_blocked={use_blocked} \\
    --ref_image={ref_image} \\
    &> {output_log}
"""


################################################################################
# Begin script
################################################################################
experiment_dir = os.path.join(EXPERIMENTS_DIR_PREFIX, VIDEO_INFO['video_name'])
if (os.path.exists(experiment_dir)):
	print experiment_dir, "already exists."
	print "WARNING. (remove the dir if you want to rerun)"
	print

try:
	os.makedirs(experiment_dir)
except:
	print experiment_dir, "already exists"

os.chdir(experiment_dir)

try:
	os.mkdir(TRAIN_DIRNAME)
except:
	print TRAIN_DIRNAME, "already exists"

################################################################################
# get training data for the optimizer and get ground truth for accuracy
################################################################################
print "preparing the training data (for optimizer) and getting ground truth"
object_id = VIDEO_INFO['object_id']
pipelines = VIDEO_INFO['pipelines']
video_name = VIDEO_INFO['video_name']
video_postfix = VIDEO_INFO['video_postfix']
VAL_START_IDX = VIDEO_INFO['val_start']
VAL_LEN = VIDEO_INFO['val_len']
VAL_END_IDX = VAL_START_IDX + VAL_LEN
TEST_START_IDX = VIDEO_INFO['test_start']
TEST_LEN = VIDEO_INFO['test_len']
TEST_END_IDX = TEST_START_IDX + TEST_LEN

val_csv_filename = "val_" + str(VAL_START_IDX) + "_" + str(VAL_START_IDX + VAL_LEN) + ".csv"
val_log_filename = "val_" + str(VAL_START_IDX) + "_" + str(VAL_START_IDX + VAL_LEN) + ".log"

video_path = os.path.join(VIDEO_DIR_PREFIX, video_name + video_postfix)
cnn_avg_path = os.path.join(CNN_AVG_DIR_PREFIX, video_name + '.txt')

pipeline_paths = []
for cnn, dd in pipelines:

	dd_name = 'non_blocked_mse.src'
	if (dd is not None):
		dd_name = dd[0]
	pipeline_path = os.path.join(EXPERIMENTS_DIR_PREFIX,
	                             video_name,
	                             TRAIN_DIRNAME,
	                             cnn + '-' + dd_name)

	cnn_path = os.path.join(CNN_MODEL_DIR_PREFIX, cnn)
	pipeline_paths.append((pipeline_path, cnn_path, dd))
	try:
		os.mkdir(pipeline_path)
	except:
		print pipeline_path, "already exists"

	val_csv_path = os.path.join(
		pipeline_path,
		val_csv_filename
	)
	val_log_path = os.path.join(
		pipeline_path,
		val_log_filename
	)

	use_blocked = 0
	diff_detection_weights = '/dev/null'
	ref_image = 0
	if dd is not None:
		use_blocked = 1
		diff_detection_weights = os.path.join(DD_MEAN_DIR_PREFIX, video_name, dd[0])
		ref_image = dd[1]

	run_optimizer_script = os.path.join(pipeline_path, RUN_OPTIMIZER_SCRIPT_FILENAME)
	with open(run_optimizer_script, 'w') as f:
		script = RUN_BASH_SCRIPT.format(
			date=str(datetime.datetime.now()),
			noscope_app_path=NOSCOPE_APP_PATH,
			dd_thres=0,
			large_cnn_thres=0,
			small_cnn_lower_thres=0,
			small_cnn_upper_thres=0,
			object_id=object_id,
			skip_dd=0,
			skip_small_cnn=0,
			kskip=30,
			diff_detection_weights=diff_detection_weights,
			use_blocked=use_blocked,
			ref_image=ref_image,
			cnn_avg_path=cnn_avg_path,
			small_cnn_path=cnn_path,
			large_cnn_path=large_cnn_path,
			video_path=video_path,
			start_frame=VAL_START_IDX,
			nb_frames=VAL_LEN,
			output_csv=val_csv_path,
			output_log=val_log_path
		)

		f.write(script)

	st = os.stat(run_optimizer_script)
	os.chmod(run_optimizer_script, st.st_mode | stat.S_IEXEC)
	print 'obtaining the optimizer data for', pipeline_path

	if (not os.path.exists(val_csv_path) or NO_CACHING):
		print "GPU_NUM:", GPU_NUM
		os.system('bash ' + run_optimizer_script + ' ' + str(GPU_NUM))
	else:
		print 'WARNING: using cached results! Skipping computation.'

################################################################################
# find the best pipeline for each error rate
################################################################################
summary_file = open(OUTPUT_SUMMARY_CSV, 'w')
summary_file.write(
	'target_fn, target_fp, skip_dd, skip_small_cnn, dd, dd_thres, small_cnn, small_cnn_upper_thres, small_cnn_lower_thres, accuracy, fn, fp, num_tp, num_tn, runtime\n')
summary_file.flush()
truth_csv = os.path.join(TRUTH_DIR_PREFIX, video_name + '.csv')

test_csv_filename = "test_" + str(TEST_START_IDX) + "_" + str(TEST_START_IDX + TEST_LEN) + ".csv"
test_log_filename = 'test_' + str(TEST_START_IDX) + "_" + str(TEST_START_IDX + TEST_LEN) + '.log'

for error_rate in TARGET_ERROR_RATES:
	test_path = os.path.join(EXPERIMENTS_DIR_PREFIX, video_name, str(error_rate))
	try:
		os.mkdir(test_path)
	except:
		pass

	# find the best configuration (find the optimal params for all of them)
	params_list = []
	for pipeline_path, small_cnn_path, dd in pipeline_paths:
		params = opt.main(
			object_id,
			truth_csv, val_csv_path,
			error_rate, error_rate,
			VAL_START_IDX, VAL_END_IDX
		)

		use_blocked = 0
		diff_detection_weights = '/dev/null'
		ref_image = 0
		if dd is not None:
			use_blocked = 1
			diff_detection_weights = os.path.join(DD_MEAN_DIR_PREFIX, video_name, dd[0])
			ref_image = dd[1]

		params['pipeline_path'] = pipeline_path
		params['small_cnn_path'] = small_cnn_path
		params['dd_path'] = diff_detection_weights
		params['dd_ref_index'] = ref_image
		params['use_blocked'] = use_blocked
		params_list.append(params)

	best_params = sorted(params_list, key=lambda x: x['optimizer_cost'])[0]
	# NOTE FIXME TODO
	# THIS IS A HACK
	best_params['threshold_skip_distance'] = 30

	# run the actual experiment
	test_csv_path = os.path.join(
		test_path,
		test_csv_filename
	)
	test_log_path = os.path.join(
		test_path,
		test_log_filename
	)

	best_params_file = os.path.join(test_path, 'params.json')
	with open(best_params_file, 'w') as f:
		f.write(json.dumps(best_params, sort_keys=True, indent=4))

	run_test_script = os.path.join(test_path, RUN_TEST_SCRIPT_FILENAME)
	with open(run_test_script, 'w') as f:
		script = RUN_BASH_SCRIPT.format(
			date=str(datetime.datetime.now()),
			noscope_app_path=NOSCOPE_APP_PATH,
			dd_thres=best_params['threshold_diff'],
			large_cnn_thres=0,
			small_cnn_lower_thres=best_params['threshold_lower_small_cnn'],
			small_cnn_upper_thres=best_params['threshold_upper_small_cnn'],
			skip_dd=int(best_params['skip_dd']),
			skip_small_cnn=int(best_params['skip_small_cnn']),
			kskip=best_params['threshold_skip_distance'],
			ref_image=int(best_params['dd_ref_index'] / best_params['threshold_skip_distance']),
			diff_detection_weights=best_params['dd_path'],
			use_blocked=best_params['use_blocked'],
			small_cnn_path=best_params['small_cnn_path'],
			large_cnn_path=large_cnn_path,
			cnn_avg_path=cnn_avg_path,
			video_path=video_path,
			object_id=object_id,
			start_frame=TEST_START_IDX,
			nb_frames=TEST_LEN,
			output_csv=test_csv_path,
			output_log=test_log_path
		)

		f.write(script)

	st = os.stat(run_test_script)
	os.chmod(run_test_script, st.st_mode | stat.S_IEXEC)
	print 'running experiment: {} {}'.format(error_rate, test_path)
	if (not os.path.exists(test_csv_path) or NO_CACHING):
		print "GPU_NUM:", GPU_NUM
		os.system('bash ' + run_test_script + ' ' + str(GPU_NUM))
	else:
		print 'WARNING: using cached results! Skipping computation.'

	# compute the actual accuracy
	accuracy = acc.main(object_id,
	                    TEST_START_IDX, TEST_END_IDX,
	                    truth_csv, test_csv_path)
	with open(os.path.join(test_path, 'accuracy.json'), 'w') as f:
		f.write(json.dumps(accuracy, sort_keys=True, indent=4))

	summary_file.write(
		'{target_fn}, {target_fp}, {skip_dd}, {skip_small_cnn}, {dd}, {dd_thres}, {small_cnn}, {small_cnn_upper_thres}, {small_cnn_lower_thres}, {acc}, {fn}, {fp}, {tp}, {tn}, {runtime}\n'.format(
			target_fn=error_rate,
			target_fp=error_rate,
			skip_dd=best_params['skip_dd'],
			skip_small_cnn=best_params['skip_small_cnn'],
			dd=best_params['dd_path'],
			dd_thres=best_params['threshold_diff'],
			small_cnn=best_params['small_cnn_path'],
			small_cnn_upper_thres=best_params['threshold_upper_small_cnn'],
			small_cnn_lower_thres=best_params['threshold_lower_small_cnn'],
			acc=accuracy['accuracy'],
			fn=accuracy['false_negative'],
			fp=accuracy['false_positive'],
			tp=accuracy['num_true_positives'],
			tn=accuracy['num_true_negatives'],
			runtime=accuracy['runtime']
		))
	summary_file.flush()

summary_file.close()
