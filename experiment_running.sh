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

# 5 fold training for norm and scaling
python 5fold_training.py --arch inception --gpu_num 0 --max_len 10000 --batch_tfms sc --scale_type nearest_exact --scale 0.5 


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


# inception cpsc
python scaling_experiment.py --arch inception --gpu_num 0 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.1 --dataset ChapmanShaoxing &

python scaling_experiment.py --arch inception --gpu_num 1 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.25 --dataset ChapmanShaoxing &

python scaling_experiment.py --arch inception --gpu_num 2 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.5 --dataset ChapmanShaoxing &

python scaling_experiment.py --arch inception --gpu_num 3 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 0.75 --dataset ChapmanShaoxing &

python scaling_experiment.py --arch inception --gpu_num 4 --max_len 5000 --batch_tfms sc n --scale_type nearest-exact --scale 1 --dataset ChapmanShaoxing & 
