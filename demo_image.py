import os
import argparse
import matplotlib.pyplot as plt
import torch
from PIL import Image
from src.model import get_model
import torchvision.transforms.functional as tf
from util.tools import load_state_dict
from util import get_cfg


def norm_input(x, patch_size=32):
    x = tf.to_tensor(x)
    _, height, width = x.shape

    target_height = (height + patch_size - 1) // patch_size * patch_size
    target_width = (width + patch_size - 1) // patch_size * patch_size

    pad_left = max(0, (target_width - width) // 2)
    pad_right = max(0, target_width - width - pad_left)
    pad_top = max(0, (target_height - height) // 2)
    pad_bottom = max(0, target_height - height - pad_top)

    padding = [pad_left, pad_top, pad_right, pad_bottom]
    x = tf.pad(x, padding, padding_mode='reflect')
    return x.unsqueeze(0), padding


def norm_output(x):
    return x.clip(0.0, 1.0).squeeze(0)


def remove_padding(tensor, padding=None):
    if padding is not None:
        pad_left, pad_top, pad_right, pad_bottom = padding
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            _, _, h, w = tensor.shape
            tensor = tensor[:, :, pad_top:h - pad_bottom, pad_left:w - pad_right]
    return tensor


def show_comparison(noise_image, enhance_image):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(noise_image)
    plt.title('Input')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(enhance_image)
    plt.title('Enhance')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--cfg_path', type=str, default='configs/test/HCLR-Net-Test.yaml')
    args = parser.parse_args()

    cfg = get_cfg(cfg_path=args.cfg_path, mode='test')

    model = get_model(cfg)
    load_state_dict(model, checkpoint_path=cfg.tester.resume_ckpt)
    model = model.cuda()
    model.eval()

    output_dir = os.path.join('enhanced_images', cfg.tester.save_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:
        img_path = input("please input the path of image: ->")
        while not os.path.exists(img_path):
            img_path = input("The input image path does not exist! Enter again: ->")

        output_path = f'{output_dir}/' + os.path.basename(str(img_path)).split(".")[0] + '_enhanced.png'

        input_pil = Image.open(img_path).convert("RGB")
        input_image, padding = norm_input(input_pil)
        input_image = input_image.cuda()

        with torch.no_grad():
            output_image = model(input_image)

        output_image = remove_padding(output_image, padding)
        output_image = norm_output(output_image)

        output_pil = tf.to_pil_image(output_image)
        output_pil.save(output_path)

        show_comparison(input_pil, output_pil)
