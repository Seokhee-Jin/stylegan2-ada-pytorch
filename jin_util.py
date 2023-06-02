import torch
import pickle
from PIL import Image
import os
import time
from typing import Tuple
import numpy as np
import torch
import warnings
import pickle

MODEL_PKL = '/home/hail/PycharmProjects/seokhee_jin/data/output/00003-men_0516_dataset-auto1-gamma0.82-bgcfnc-resumeffhq256/network-snapshot-001000.pkl'
MODEL_PKL = '/media/jin/Jin_USB_1/result_women/pkl/network-snapshot-004720.pkl'

### load trained model ###
with open(MODEL_PKL, 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

### generatge img from noise ###
def gen_img(truncation_psi=0.5, truncation_cutoff=8) -> Tuple[torch.Tensor, Image.Image]:
    z = torch.randn([1, G.z_dim]).cuda()  # latent codes
    c = None  # class labels (not used in this example)
    w = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
    img = G.synthesis(w, noise_mode='const', force_fp32=True)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = Image.fromarray(img[0].cpu().numpy(), 'RGB')
    return w, img

def gen_img_from_noise(z) -> Tuple[torch.Tensor, Image.Image]:
    warnings.warn("deprecated", DeprecationWarning)
    c = None  # class labels (not used in this example)
    img = G(z, c)  # NCHW, float32, dynamic range [-1, +1]
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = Image.fromarray(img[0].cpu().numpy(), 'RGB')
    return img

def gen_img_from_w(w) -> Tuple[torch.Tensor, Image.Image]:
    if isinstance(w, np.ndarray):
        w = torch.Tensor(w).to('cuda')
    img = G.synthesis(w, noise_mode='const', force_fp32=True)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = Image.fromarray(img[0].cpu().numpy(), 'RGB')
    return img



### show img ###

# img1.show()
# img1_5.show()
# img2.show()


### save img ###
def save_img(img: Image.Image, save_dir: str, img_name: str = None):
    os.makedirs(save_dir, exist_ok=True)
    if img_name is None:
        img_name = str(time.time()) + ".png"
    full_path = os.path.join(save_dir, img_name)
    img.save(full_path)


### interpolation ####
def interpolate_two(w1, w2, ratio):
    assert ratio >= 0 and ratio <= 1

    if isinstance(w1, np.ndarray):
        w1 = torch.Tensor(w1).to('cuda')

    if isinstance(w2, np.ndarray):
        w2 = torch.Tensor(w2).to('cuda')

    return (w1*ratio + w2*(1-ratio))

if __name__ == "__name__":
    # save_img(img1, "./generated_images")
    # save_img(img1_5, "./generated_images")
    # save_img(img2, "./generated_images")

    w, img = gen_img()
    img.show()

    #### intermediate latent space w ###
    npz = np.load("/home/hail/PycharmProjects/seokhee_jin/data/output/projected/projected_w1000.npz")
    w1 = npz['w']

    img1 = gen_img_from_w(w1)
    img1.show()

    w2, img2 = gen_img()
    img2.show()
    save_img(img2, "/home/hail/PycharmProjects/seokhee_jin/data/output", 'fromz.png')

    RATIO = 0.0
    temp = gen_img_from_w(interpolate_two(w1, w2, RATIO))
    save_img(temp, "/home/hail/PycharmProjects/seokhee_jin/data/output", str(RATIO) + ".png")
