import os
import torch
import numpy as np
from util.tools import get_timepc, trans_state_dict
from util.metric import IQAMetrics
from .basic_trainer import BasicTrainer


class HCLRTrainer(BasicTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_psnr = float("-inf")
        self.is_best_psnr = False
        self.metrics = IQAMetrics()

    def set_input(self, inputs):
        self.input = inputs['img'].to(self.local_rank)
        self.target = inputs['target'].to(self.local_rank)
        self.unpair = inputs['unpair_img'].to(self.local_rank)

    def get_loss(self):
        return self.loss_fn(self.output, self.target, self.unpair)

    def update_cur_log(self):
        self.lr = self.optim.state_dict()['param_groups'][0]['lr']

        self.cur_log = "[Train: {:.2f}% | {}/{}]\t[Loss: {:.4f}\tL1: {:.4f}\tperc: {:.4f}\tperc: {:.4f}]\t[LR: {:.6f}]".format(
            100*self.iter_now/self.total_iter, self.iter_now, self.total_iter, self.loss['loss'], self.loss['l1'], self.loss['perc'], self.loss['ucr'], self.lr)

    @torch.no_grad()
    def test_model_multi(self):
        t_s = get_timepc()
        self.reset(is_train=False)
        self.check_bn()

        psnr_list, ssim_list = [], []
        for k, v in self.test_loader.items():
            psnr, ssim = 0.0, 0.0
            _t_s = get_timepc()
            for idx, batch_data in enumerate(v):
                self.set_input(batch_data)
                self.forward()

                _PSNR, _SSIM = self.metrics.get_metrics(self.output, self.target)
                psnr += _PSNR
                ssim += _SSIM

            psnr = psnr / len(v)
            ssim = ssim / len(v)

            _t_e = get_timepc()
            t_cost = _t_e - _t_s
            self.cur_logs = "[{:^15}]\t[Time Cost:{:.3f}s]\t[PSNR:{:.4f} | SSIM:{:.4f}]".format(k, t_cost, psnr, ssim)
            self.logger.log_msg(self.cur_logs) if self.master else None
            psnr_list.append(psnr)
            ssim_list.append(ssim)

        psnr = np.mean(psnr_list)
        ssim = np.mean(ssim_list)

        t_e = get_timepc()
        t_cost = t_e - t_s
        self.cur_logs = "Test Done!\tTime Cost:{:.3f}s\tPSNR:{:.4f}\tSSIM:{:.4f}".format(t_cost, psnr, ssim)
        self.logger.log_msg(self.cur_logs) if self.master else None
        if self.best_psnr < psnr:
            self.best_psnr = psnr
            self.is_best_psnr = True

    def save_ckpt(self):
        if self.master:
            ckpt_infos = {
                "model": trans_state_dict(self.model.state_dict(), is_dist=False),
                "optimizer": self.optim.state_dict(),
                "iter": self.iter_now,
                "epoch": self.epoch_now,
            }
            print()
            dir_name = os.path.join(self.cfg.trainer.ckpt_dir, str(self.cfg.model.name), self.cfg.task_start_time)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            if self.epoch_now % self.cfg.trainer.save_freq == 0:
                base_name = "latest_ckpt.pth"
                save_path = os.path.join(str(dir_name), base_name)
                torch.save(ckpt_infos, save_path)
                self.logger.log_msg("checkpoint saved to {}".format(save_path)) if self.master else None
            if self.is_best_psnr:
                base_name = "best_psnr_ckpt.pth"
                save_path = os.path.join(str(dir_name), base_name)
                torch.save(ckpt_infos, save_path)
                self.is_best_psnr = False
                self.logger.log_msg("checkpoint saved to {}".format(save_path)) if self.master else None
