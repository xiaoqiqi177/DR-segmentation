# Training Phase
CUDA_VISIBLE_DEVICES=0 python train.py

# Test Phase
CUDA_VISIBLE_DEVICES=0 python test_npy.py
python test_ap.py
