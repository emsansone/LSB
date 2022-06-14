rm -r __pycache__
rm -r results_*

n_steps=120000
n_samples=500
n_test_samples=16 # Batch size for sampler
gt_steps=120000

# Params for LSB 2
lr=1e-2
lsb_hidden=10

data='mnist'
save_dir='results_'$data
n_hidden=250
n_visible=784
batch_size=28   # Batch size for RBM training
print_every=500 

python3 rbm_sample.py --data $data --save_dir $save_dir --n_steps $n_steps --n_samples $n_samples --n_test_samples $n_test_samples --gt_steps $gt_steps --n_hidden $n_hidden --n_visible $n_visible --batch_size $batch_size --print_every $print_every --lr $lr --lsb_hidden $lsb_hidden
