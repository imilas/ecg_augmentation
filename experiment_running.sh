# python 10_fold_training.py --gpu_num 7 --max_len 10000 --batch_tfms sc &
# python 10_fold_training.py --gpu_num 6 --max_len 10000 --batch_tfms sc n & 
# python 10_fold_training.py --gpu_num 5 --max_len 10000 --batch_tfms sc n bp &
# python 10_fold_training.py --gpu_num 4 --max_len 10000 --batch_tfms sc n bp sh &

# python 10_fold_training.py --arch minirocket --gpu_num 6 --max_len 6000
# python 10_fold_training.py --arch minirocket --gpu_num 6 --max_len 6000 --batch_tfms sc &
# python 10_fold_training.py --arch minirocket --gpu_num 5 --max_len 6000 --batch_tfms sc n & 
# python 10_fold_training.py --arch minirocket --gpu_num 4 --max_len 6000 --batch_tfms sc n bp &
# python 10_fold_training.py --arch minirocket --gpu_num 3 --max_len 6000 --batch_tfms sc n bp sh &

python 10_fold_training.py --gpu_num 7 --max_len 5000 --dataset ChapmanShaoxing
python 10_fold_training.py --gpu_num 7 --max_len 5000 --dataset ChapmanShaoxing --batch_tfms sc &
python 10_fold_training.py --gpu_num 6 --max_len 5000 --dataset ChapmanShaoxing --batch_tfms sc n & 
python 10_fold_training.py --gpu_num 5 --max_len 5000 --dataset ChapmanShaoxing --batch_tfms sc n bp &
python 10_fold_training.py --gpu_num 4 --max_len 5000 --dataset ChapmanShaoxing --batch_tfms sc n bp sh &
