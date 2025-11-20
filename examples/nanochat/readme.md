
python  -m train_loop.train ./examples/nanochat/nanochat.yml # single 80GB GPU

python  -m train_loop.train ./examples/nanochat/nanochat.yml train.batch_size=1 # single 16GB GPU

CUDA_VISIBLE_DEVICES=1,2  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" python  -m train_loop.train ./examples/nanochat/nanochat.yml train.n_gpus=2 train.use_ddp=True 