import os

if __name__ == "__main__":
    for seed in range(5):
        os.system(f'python train.py VAE --gpus 1 --gradient_clip_val 1.0 --seed {seed}')