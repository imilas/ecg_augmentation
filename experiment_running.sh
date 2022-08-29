# python 10_fold_training.py --gpu_num 7 --max_len 10000 --batch_tfms sc &
# python 10_fold_training.py --gpu_num 6 --max_len 10000 --batch_tfms sc n & 
# python 10_fold_training.py --gpu_num 5 --max_len 10000 --batch_tfms sc n bp &
# python 10_fold_training.py --gpu_num 4 --max_len 10000 --batch_tfms sc n bp sh &

# python 10_fold_training.py --arch minirocket --gpu_num 6 --max_len 6000
# python 10_fold_training.py --arch minirocket --gpu_num 6 --max_len 6000 --batch_tfms sc &
# python 10_fold_training.py --arch minirocket --gpu_num 5 --max_len 6000 --batch_tfms sc n & 
# python 10_fold_training.py --arch minirocket --gpu_num 4 --max_len 6000 --batch_tfms sc n bp &
# python 10_fold_training.py --arch minirocket --gpu_num 3 --max_len 6000 --batch_tfms sc n bp sh &

# python 10_fold_training.py --gpu_num 5 --max_len 5000 --dataset ChapmanShaoxing --arch inception
# python 10_fold_training.py --gpu_num 4 --max_len 5000 --dataset ChapmanShaoxing --arch inception --batch_tfms sc &
# python 10_fold_training.py --gpu_num 6 --max_len 5000 --dataset ChapmanShaoxing --arch inception --batch_tfms sc n & 
# python 10_fold_training.py --gpu_num 7 --max_len 5000 --dataset ChapmanShaoxing --arch inception --batch_tfms sc n bp &
# python 10_fold_training.py --gpu_num 4 --max_len 5000 --dataset ChapmanShaoxing --arch inception --batch_tfms sc n bp sh &

# python 10_fold_training.py --arch minirocket --gpu_num 6 --max_len 5000 --dataset ChapmanShaoxing --cv_range 8 9 
# python 10_fold_training.py --arch minirocket  --gpu_num 7 --max_len 5000 --dataset ChapmanShaoxing --batch_tfms sc &
# python 10_fold_training.py --arch minirocket  --gpu_num 6 --max_len 5000 --dataset ChapmanShaoxing  --batch_tfms sc n & 
# python 10_fold_training.py --arch minirocket  --gpu_num 5 --max_len 5000 --dataset ChapmanShaoxing  --batch_tfms sc n bp &
# python 10_fold_training.py --arch minirocket --gpu_num 4 --max_len 5000 --dataset ChapmanShaoxing  --batch_tfms sc n bp sh &


### Scaling Experiments ####
# inception cpsc
python scaling_experiment.py --arch inception --gpu_num 0 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.1 &
python scaling_experiment.py --arch inception --gpu_num 1 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.25 &
python scaling_experiment.py --arch inception --gpu_num 2 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.5 &
python scaling_experiment.py --arch inception --gpu_num 3 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.75 &
python scaling_experiment.py --arch inception --gpu_num 3 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 1 &

# minirocket cpsc
python scaling_experiment.py --arch minirocket --gpu_num 0 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.1 &
python scaling_experiment.py --arch minirocket --gpu_num 1 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.25 &
python scaling_experiment.py --arch minirocket --gpu_num 2 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.5 &
python scaling_experiment.py --arch minirocket --gpu_num 3 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.75 &
python scaling_experiment.py --arch minirocket --gpu_num 4 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 1 &

# xresnet101 cpsc
python scaling_experiment.py --arch xresnet1d101 --gpu_num 0 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.1 &
python scaling_experiment.py --arch xresnet1d101 --gpu_num 0 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.25  &
python scaling_experiment.py --arch xresnet1d101 --gpu_num 1 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.5  &
python scaling_experiment.py --arch xresnet1d101 --gpu_num 2 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.75 &
python scaling_experiment.py --arch xresnet1d101 --gpu_num 3 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 1 &


# inception chapman
python scaling_experiment.py --arch inception --gpu_num 0 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.1 --dataset ChapmanShaoxing &
python scaling_experiment.py --arch inception --gpu_num 1 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.25 --dataset ChapmanShaoxing &
python scaling_experiment.py --arch inception --gpu_num 2 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.5 --dataset ChapmanShaoxing &
python scaling_experiment.py --arch inception --gpu_num 3 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.75 --dataset ChapmanShaoxing &
python scaling_experiment.py --arch inception --gpu_num 4 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 1 --dataset ChapmanShaoxing & 


# minirocket chapman

python scaling_experiment.py --arch minirocket --gpu_num 0 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.1 --dataset ChapmanShaoxing &
python scaling_experiment.py --arch minirocket --gpu_num 0 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.25 --dataset ChapmanShaoxing &
python scaling_experiment.py --arch minirocket --gpu_num 1 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.5 --dataset ChapmanShaoxing &
python scaling_experiment.py --arch minirocket --gpu_num 2 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.75 --dataset ChapmanShaoxing &
python scaling_experiment.py --arch minirocket --gpu_num 3 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 1 --dataset ChapmanShaoxing & 


# xresnet101 chapman
python scaling_experiment.py --arch xresnet1d101 --gpu_num 0 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.1 --dataset ChapmanShaoxing &
python scaling_experiment.py --arch xresnet1d101 --gpu_num 0 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.25 --dataset ChapmanShaoxing & 
python scaling_experiment.py --arch xresnet1d101 --gpu_num 1 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.5 --dataset ChapmanShaoxing & 
python scaling_experiment.py --arch xresnet1d101 --gpu_num 2 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.75 --dataset ChapmanShaoxing & 
python scaling_experiment.py --arch xresnet1d101 --gpu_num 3 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 1 --dataset ChapmanShaoxing & 


# inception PTBXL
python scaling_experiment.py --arch inception --gpu_num 0 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.1 --dataset PTBXL &
python scaling_experiment.py --arch inception --gpu_num 0 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.25 --dataset PTBXL &
python scaling_experiment.py --arch inception --gpu_num 1 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.5 --dataset PTBXL &
python scaling_experiment.py --arch inception --gpu_num 2 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.75 --dataset PTBXL &
python scaling_experiment.py --arch inception --gpu_num 3 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 1 --dataset PTBXL &


# minirocket PTBXL
python scaling_experiment.py --arch minirocket --gpu_num 0 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.1 --dataset PTBXL &
python scaling_experiment.py --arch minirocket --gpu_num 0 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.25 --dataset PTBXL &
python scaling_experiment.py --arch minirocket --gpu_num 1 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.5 --dataset PTBXL &
python scaling_experiment.py --arch minirocket --gpu_num 2 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.75 --dataset PTBXL &
python scaling_experiment.py --arch minirocket --gpu_num 3 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 1 --dataset PTBXL &



# xresnet101 PTBXL
python scaling_experiment.py --arch xresnet1d101 --gpu_num 0 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.1 --dataset PTBXL &
python scaling_experiment.py --arch xresnet1d101 --gpu_num 0 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.25 --dataset PTBXL &
python scaling_experiment.py --arch xresnet1d101 --gpu_num 1 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.5 --dataset PTBXL &
python scaling_experiment.py --arch xresnet1d101 --gpu_num 2 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.75 --dataset PTBXL &
python scaling_experiment.py --arch xresnet1d101 --gpu_num 3 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 1 --dataset PTBXL &


##### bandpass experiments 
# cpsc
python bandpass_experiment.py --arch inception --gpu_num 0 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 30 &
python bandpass_experiment.py --arch inception --gpu_num 1 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 50 &
python bandpass_experiment.py --arch inception --gpu_num 2 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 100 &


python bandpass_experiment.py --arch minirocket --gpu_num 3 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 30 &
python bandpass_experiment.py --arch minirocket --gpu_num 3 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 50 &
python bandpass_experiment.py --arch minirocket --gpu_num 3 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 100 &


python bandpass_experiment.py --arch xresnet1d101 --gpu_num 0 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 30 &
python bandpass_experiment.py --arch xresnet1d101 --gpu_num 1 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 50 &
python bandpass_experiment.py --arch xresnet1d101 --gpu_num 2 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 100 &

# chapman
python bandpass_experiment.py --arch inception --gpu_num 0 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 30 --dataset ChapmanShaoxing &
python bandpass_experiment.py --arch inception --gpu_num 1 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 50 --dataset ChapmanShaoxing &
python bandpass_experiment.py --arch inception --gpu_num 2 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 100 --dataset ChapmanShaoxing &


python bandpass_experiment.py --arch minirocket --gpu_num 3 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 30 --dataset ChapmanShaoxing &
python bandpass_experiment.py --arch minirocket --gpu_num 3 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 50 --dataset ChapmanShaoxing &
python bandpass_experiment.py --arch minirocket --gpu_num 3 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 100 --dataset ChapmanShaoxing &


python bandpass_experiment.py --arch xresnet1d101 --gpu_num 3 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 30 --dataset ChapmanShaoxing &
python bandpass_experiment.py --arch xresnet1d101 --gpu_num 4 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 50 --dataset ChapmanShaoxing &
python bandpass_experiment.py --arch xresnet1d101 --gpu_num 5 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 100 --dataset ChapmanShaoxing &

# PTBXL

python bandpass_experiment.py --arch inception --gpu_num 0 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 30 --dataset PTBXL &
python bandpass_experiment.py --arch inception --gpu_num 1 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 50 --dataset PTBXL &
python bandpass_experiment.py --arch inception --gpu_num 2 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 100 --dataset PTBXL &


python bandpass_experiment.py --arch minirocket --gpu_num 3 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 30 --dataset PTBXL &
python bandpass_experiment.py --arch minirocket --gpu_num 3 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 50 --dataset PTBXL &
python bandpass_experiment.py --arch minirocket --gpu_num 3 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 100 --dataset PTBXL &


python bandpass_experiment.py --arch xresnet1d101 --gpu_num 0 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 30 --dataset PTBXL &
python bandpass_experiment.py --arch xresnet1d101 --gpu_num 1 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 50 --dataset PTBXL &
python bandpass_experiment.py --arch xresnet1d101 --gpu_num 2 --max_len 5000 --batch_tfms sc n bp --scale_type nearest-exact --scale 0.5 --HP 1 --LP 100 --dataset PTBXL &


### norm experiments ###

# cpsc
python norm_experiment.py --arch inception --gpu_num 1 --max_len 5000 --scale_type nearest-exact --scale 1 &
python norm_experiment.py --arch minirocket --gpu_num 2 --max_len 5000  --scale_type nearest-exact --scale 1 &
python norm_experiment.py --arch xresnet1d101 --gpu_num 4 --max_len 5000 --scale_type nearest-exact --scale 1 &

# chapman
python norm_experiment.py --arch inception --gpu_num 1 --max_len 5000 --scale_type nearest-exact --scale 1 --dataset ChapmanShaoxing & 
python norm_experiment.py --arch minirocket --gpu_num 2 --max_len 5000  --scale_type nearest-exact --scale 1 --dataset ChapmanShaoxing & 
python norm_experiment.py --arch xresnet1d101 --gpu_num 3 --max_len 5000 --scale_type nearest-exact --scale 1 --dataset ChapmanShaoxing & 

# ptbxl
python norm_experiment.py --arch inception --gpu_num 1 --max_len 5000 --scale_type nearest-exact --scale 1 --dataset PTBXL & 
python norm_experiment.py --arch minirocket --gpu_num 2 --max_len 5000  --scale_type nearest-exact --scale 1 --dataset PTBXL & 
python norm_experiment.py --arch xresnet1d101 --gpu_num 0 --max_len 5000 --scale_type nearest-exact --scale 1 --dataset PTBXL &