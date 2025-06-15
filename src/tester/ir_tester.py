import os
import torch
import numpy as np
from util.tools import get_timepc, remove_padding
from util.metric import NTIREScore
from .basic_tester import BasicTester
import torchvision


class IRTester(BasicTester):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ntir_score = NTIREScore()

    def forward(self):
        with self.autocast:
            self.output = self.model(self.input)

    def set_input(self, inputs):
        self.input = inputs['img'].to(self.local_rank)
        self.target = inputs['target'].to(self.local_rank)
        self.input_path = inputs['input_path']
        self.output_path = inputs['output_path']
        self.comparison_path = inputs['comparison_path']
        self.input_padding = inputs['input_padding']

    @torch.no_grad()
    def test(self):
        t_s = get_timepc()

        overall_logs = []
        input_psnr_list, input_ssim_list, input_lpips_list, input_dists_list, input_niqe_list, input_score_list = [], [], [], [], [], []
        output_psnr_list, output_ssim_list, output_lpips_list, output_dists_list, output_niqe_list, output_score_list = [], [], [], [], [], []

        for k, v in self.test_loader.items():
            if self.cfg.tester.save_output:
                if not os.path.exists(self.cfg.tester.save_dir + f'/output/{k}'):
                    os.makedirs(self.cfg.tester.save_dir + f'/output/{k}')
            if self.cfg.tester.save_comparison:
                if not os.path.exists(self.cfg.tester.save_dir + f'/comparison/{k}'):
                    os.makedirs(self.cfg.tester.save_dir + f'/comparison/{k}')

            input_psnr, input_ssim, input_lpips, input_dists, input_niqe, input_score = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            output_psnr, output_ssim, output_lpips, output_dists, output_niqe, output_score = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            _t_s = get_timepc()
            for idx, batch_data in enumerate(v):
                self.set_input(batch_data)
                self.forward()

                self.input = remove_padding(self.input, self.input_padding)
                self.output = remove_padding(self.output, self.input_padding)

                # enhance metrics
                output_metrics_dict = self.ntir_score.get_score(self.output, self.target)
                output_psnr += output_metrics_dict['psnr']
                output_ssim += output_metrics_dict['ssim']
                output_lpips += output_metrics_dict['lpips']
                output_dists += output_metrics_dict['dists']
                output_niqe += output_metrics_dict['niqe']
                output_score += output_metrics_dict['score']

                if self.cfg.tester.compute_input_metrics:
                    # input metrics
                    input_metrics_dict = self.ntir_score.get_score(self.input, self.target)
                    input_psnr += input_metrics_dict['psnr']
                    input_ssim += input_metrics_dict['ssim']
                    input_lpips += input_metrics_dict['lpips']
                    input_dists += input_metrics_dict['dists']
                    input_niqe += input_metrics_dict['niqe']
                    input_score += input_metrics_dict['score']

                    self.cur_logs = ("[input image path: {}]\n"
                                     "Input\t[PSNR:{:.4f}]\t[SSIM:{:.4f}]\t[LPIPS:{:.4f}]\t[DISTS:{:.4f}]\t[NIQE:{:.4f}]\t[Score:{:.4f}]\n"
                                     "Enhanced\t[PSNR:{:.4f}]\t[SSIM:{:.4f}]\t[LPIPS:{:.4f}]\t[DISTS:{:.4f}]\t[NIQE:{:.4f}]\t[Score:{:.4f}]").format(
                        self.input_path[0],
                        input_metrics_dict['psnr'], input_metrics_dict['ssim'], input_metrics_dict['lpips'],
                        input_metrics_dict['dists'], input_metrics_dict['niqe'], input_metrics_dict['score'],
                        output_metrics_dict['psnr'], output_metrics_dict['ssim'], output_metrics_dict['lpips'],
                        output_metrics_dict['dists'],
                        output_metrics_dict['niqe'], output_metrics_dict['score'])
                else:
                    self.cur_logs = ("[input image path: {}]\n"
                                     "Enhanced\t[PSNR:{:.4f}]\t[SSIM:{:.4f}]\t[LPIPS:{:.4f}]\t[DISTS:{:.4f}]\t[NIQE:{:.4f}]\t[Score:{:.4f}]").format(
                        self.input_path[0],
                        output_metrics_dict['psnr'], output_metrics_dict['ssim'], output_metrics_dict['lpips'],
                        output_metrics_dict['dists'],
                        output_metrics_dict['niqe'], output_metrics_dict['score'])
                self.logger.log_msg(self.cur_logs) if self.master else None

                if self.cfg.tester.save_output:
                    # output_image = self.output * 0.5 + 0.5
                    output_image = self.output.clip(0.0, 1.0)
                    torchvision.utils.save_image(output_image, self.output_path[0])
                    self.cur_logs = "Save Enhanced Image to: {}".format(self.output_path[0])
                    self.logger.log_msg(self.cur_logs) if self.master else None

                if self.cfg.tester.save_comparison:
                    comparison_image = torch.cat([self.input, self.output, self.target])
                    # comparison_image = comparison_image * 0.5 + 0.5
                    comparison_image = comparison_image.clip(0.0, 1.0)
                    torchvision.utils.save_image(comparison_image, self.comparison_path[0])
                    self.cur_logs = "Save comparison to: {}".format(self.comparison_path[0])
                    self.logger.log_msg(self.cur_logs) if self.master else None

            # Enhanced
            output_psnr, output_ssim, output_lpips, output_dists, output_niqe, output_score = (
                output_psnr / len(v),
                output_ssim / len(v),
                output_lpips / len(v),
                output_dists / len(v),
                output_niqe / len(v),
                output_score / len(v)
            )
            _t_e = get_timepc()

            if self.cfg.tester.compute_input_metrics:
                # Input
                input_psnr, input_ssim, input_lpips, input_dists, input_niqe, input_score = (
                    input_psnr / len(v),
                    input_ssim / len(v),
                    input_lpips / len(v),
                    input_dists / len(v),
                    input_niqe / len(v),
                    input_score / len(v)
                )
                t_cost = _t_e - _t_s
                self.cur_logs = ("\n[{:^15}]\t[Time Cost:{:.3f}s]\n"
                                 "[{:<8}]\t[PSNR:{:.4f}]\t[SSIM:{:.4f}]\t[LPIPS:{:.4f}]\t[DISTS:{:.4f}]\t[NIQE:{:.4f}]\t[Score:{:.4f}]\n"
                                 "[{:<8}]\t[PSNR:{:.4f}]\t[SSIM:{:.4f}]\t[LPIPS:{:.4f}]\t[DISTS:{:.4f}]\t[NIQE:{:.4f}]\t[Score:{:.4f}]\n").format(
                    k, t_cost,
                    'Input', input_psnr, input_ssim, input_lpips, input_dists, input_niqe, input_score,
                    'Enhanced', output_psnr, output_ssim, output_lpips, output_dists, output_niqe, output_score)
            else:
                t_cost = _t_e - _t_s
                self.cur_logs = ("\n[{:^15}]\t[Time Cost:{:.3f}s]\n"
                                 "[{:<8}]\t[PSNR:{:.4f}]\t[SSIM:{:.4f}]\t[LPIPS:{:.4f}]\t[DISTS:{:.4f}]\t[NIQE:{:.4f}]\t[Score:{:.4f}]\n").format(
                    k, t_cost,
                    'Enhanced', output_psnr, output_ssim, output_lpips, output_dists, output_niqe, output_score)
            overall_logs.append(self.cur_logs)

            output_psnr_list.append(output_psnr)
            output_ssim_list.append(output_ssim)
            output_lpips_list.append(output_lpips)
            output_dists_list.append(output_dists)
            output_niqe_list.append(output_niqe)
            output_score_list.append(output_score)

            if self.cfg.tester.compute_input_metrics:
                input_psnr_list.append(input_psnr)
                input_ssim_list.append(input_ssim)
                input_lpips_list.append(input_lpips)
                input_dists_list.append(input_dists)
                input_niqe_list.append(input_niqe)
                input_score_list.append(input_score)

        output_psnr = np.mean(output_psnr_list)
        output_ssim = np.mean(output_ssim_list)
        output_lpips = np.mean(output_lpips_list)
        output_dists = np.mean(output_dists_list)
        output_niqe = np.mean(output_niqe_list)
        output_score = np.mean(output_score_list)

        if self.cfg.tester.compute_input_metrics:
            input_psnr = np.mean(input_psnr_list)
            input_ssim = np.mean(input_ssim_list)
            input_lpips = np.mean(input_lpips_list)
            input_dists = np.mean(input_dists_list)
            input_niqe = np.mean(input_niqe_list)
            input_score = np.mean(input_score_list)

            t_e = get_timepc()
            t_cost = t_e - t_s
            self.cur_logs = ("Test Done!\tTime Cost:{:.3f}s\n"
                             "{:<8}\tPSNR:{:.4f}\tSSIM:{:.4f}\tLPIPS:{:.4f}\tDISTS:{:.4f}\tNIQE:{:.4f}\tScore:{:.4f}\n"
                             "{:<8}\tPSNR:{:.4f}\tSSIM:{:.4f}\tLPIPS:{:.4f}\tDISTS:{:.4f}\tNIQE:{:.4f}\tScore:{:.4f}").format(
                t_cost,
                'Input', input_psnr, input_ssim, input_lpips, input_dists, input_niqe, input_score,
                'Enhanced', output_psnr, output_ssim, output_lpips, output_dists, output_niqe, output_score)
        else:
            t_e = get_timepc()
            t_cost = t_e - t_s
            self.cur_logs = ("Test Done!\tTime Cost:{:.3f}s\n"
                             "{:<8}\tPSNR:{:.4f}\tSSIM:{:.4f}\tLPIPS:{:.4f}\tDISTS:{:.4f}\tNIQE:{:.4f}\tScore:{:.4f}\n").format(
                t_cost,
                'Enhanced', output_psnr, output_ssim, output_lpips, output_dists, output_niqe, output_score)

        overall_logs.append(self.cur_logs)
        for log in overall_logs:
            self.logger.log_msg(log) if self.master else None
