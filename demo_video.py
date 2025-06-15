import os
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from src.model import get_model
import torchvision.transforms.functional as tf
from util.tools import load_state_dict, get_timepc
from util import get_cfg
from torch.amp import autocast


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
    x = x * 255.0
    x = x.clip(0.0, 255.0).squeeze(0).permute(1, 2, 0).detach().cpu()
    x = np.array(x, dtype=np.uint8)
    return x


def remove_padding(tensor, padding=None):
    if padding is not None:
        pad_left, pad_top, pad_right, pad_bottom = padding
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            _, _, h, w = tensor.shape
            tensor = tensor[:, :, pad_top:h - pad_bottom, pad_left:w - pad_right]
    return tensor


def main(cfg, comparison):
    model = get_model(cfg)
    load_state_dict(model, checkpoint_path=cfg.tester.resume_ckpt)
    model = model.cuda()
    model.eval()

    output_dir = os.path.join('enhanced_videos', cfg.tester.save_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:
        video_path = input("please input the path of video: ->")
        while not os.path.exists(video_path):
            video_path = input("The input video path does not exist! Enter again: ->")

        t_s_video = get_timepc()
        print("Start enhance video!")
        cap = cv2.VideoCapture(video_path)
        frame_index = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编解码器
        output_path = f'{output_dir}/' + os.path.basename(str(video_path)).split(".")[0] + '_enhanced.mp4'
        
        if comparison:
            if height > width:
                output_width = width * 2
                output_height = height
            else:
                output_width = width
                output_height = height * 2
        else:
            output_width = width
            output_height = height
            
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        out.set(cv2.CAP_PROP_BITRATE, value=4096000)

        if not out.isOpened():
            print("Error: Could not open the output video file for writing.")
            cap.release()
            return

        while cap.isOpened():
            t_s_cost = get_timepc()
            frame_index += 1

            t_s_data = get_timepc()
            ret, input_np = cap.read()
            if not ret:
                break

            input_pil = Image.fromarray(input_np)
            input_image, padding = norm_input(input_pil)
            input_image = input_image.cuda()
            t_e_data = get_timepc()
            data_time = t_e_data - t_s_data

            t_s_inference = get_timepc(cuda_synchronize=True)
            with autocast('cuda'):
                with torch.no_grad():
                    output_image = model(input_image)
            t_e_inference = get_timepc(cuda_synchronize=True)
            inference_time = t_e_inference -t_s_inference

            t_s_save = get_timepc()
            output_image = remove_padding(output_image, padding)
            output_np = norm_output(output_image)

            if comparison:
                if height > width:
                    output_frame = cv2.hconcat([input_np, output_np])
                else:
                    output_frame = cv2.vconcat([input_np, output_np])
            else:
                output_frame = output_np

            out.write(output_frame)
            t_e_save = get_timepc()
            save_time = t_e_save - t_s_save

            t_e_cost = get_timepc()
            cost_time = t_e_cost - t_s_cost
            print("[Frame index:{} | resolution:{}*{}]\t[time cost:{:.4f}s | data time:{:.4f}s | inference time:{:.4f}s | save time:{:.4f}s]".format(frame_index, height, width, cost_time, data_time, inference_time, save_time))

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        t_e_video = get_timepc()
        video_time = t_e_video - t_s_video
        print("Finish enhance video! Time Cost:{:.4f}s".format(video_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg_path', type=str, default='configs/test/HCLR-Net-Test.yaml')
    parser.add_argument('--comparison', action='store_true')
    args = parser.parse_args()
    cfg = get_cfg(cfg_path=args.cfg_path, mode='test')
    
    main(cfg=cfg, comparison=args.comparison)
