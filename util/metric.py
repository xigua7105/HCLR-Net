import torch
import pyiqa
import torch.nn.functional as F


def norm_data(x):
    # [-1, 1] -> [0, 255]
    return x * 127.5 + 127.5


def norm_data1(x):
    # [-1, 1] -> [0, 1]
    return x * 0.5 + 0.5


def topk_accuracy(output, target, top_k=(1, 5)):
    max_k = max(top_k)
    _, pred = output.topk(max_k, 1, True, True)
    correct = pred.eq(target.view(target.size(0), -1).expand_as(pred))
    return [correct[:, :k].sum().item() for k in top_k]


def psnr_pyiqa(output, target):
    output = norm_data1(output).clip(0, 1)
    target = norm_data1(target).clip(0, 1)

    iqa_psnr = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(output.device)
    return iqa_psnr(output, target).item()


def ssim_pyiqa(output, target):
    output = norm_data1(output).clip(0, 1)
    target = norm_data1(target).clip(0, 1)

    iqa_ssim = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(output.device)
    return iqa_ssim(output, target).item()


class IQAMetrics:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.iqa_ssim = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(self.device)
        self.iqa_psnr = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(self.device)

    def get_metrics(self, output, target):
        output = output.clip(0, 1)
        target = target.clip(0, 1)

        _PSNR = self.iqa_psnr(output, target).item()
        _SSIM = self.iqa_ssim(output, target).item()

        return _PSNR, _SSIM


class NTIREScore:
    def __init__(self):
        self.psnr_range = [0, 50]
        self.ssim_range = [0.5, 1]
        self.lpips_range = [0, 1]
        self.dists_range = [0, 1]
        self.niqe_range = [0, 1]

        self.psnr_weight = 20
        self.ssim_weight = 15
        self.lpips_weight = 20
        self.dists_weight = 40
        self.niqe_weight = 30

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.iqa_ssim = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(self.device)
        self.iqa_psnr = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(self.device)
        self.iqa_lpips = pyiqa.create_metric('lpips', device=self.device)
        self.iqa_dists = pyiqa.create_metric('dists').to(self.device)
        self.iqa_niqe = pyiqa.create_metric('niqe').to(self.device)

    def norm_score(self, score, data_range, data_weight=1):
        return data_weight * (score - data_range[0]) / (data_range[1] - data_range[0])

    def get_score(self, output, target):
        output = output.clip(0, 1)
        target = target.clip(0, 1)

        output = output[..., 4:-4, 4:-4]
        target = target[..., 4:-4, 4:-4]

        _PSNR = self.iqa_psnr(output, target).item()
        _SSIM = self.iqa_ssim(output, target).item()
        _LPIPS = self.iqa_lpips(output, target).item()
        _DISTS = self.iqa_dists(output, target).item()
        _NIQE = self.iqa_niqe(output).item()

        metrics_dict = {'psnr': _PSNR, 'ssim': _SSIM, 'lpips': _LPIPS, 'dists': _DISTS, 'niqe': _NIQE}

        psnr_score = self.norm_score(_PSNR, self.psnr_range, self.psnr_weight)
        ssim_score = self.norm_score(_SSIM, self.ssim_range, self.ssim_weight)
        lpips_score = self.norm_score(1 - _LPIPS * 2.5, self.lpips_range, self.lpips_weight)
        dists_score = self.norm_score(1 - _DISTS / 0.3, self.dists_range, self.dists_weight)
        niqe_score = self.norm_score(1 - _NIQE * 0.1, self.niqe_range, self.niqe_weight)

        final_score = psnr_score + ssim_score + lpips_score + dists_score + niqe_score
        metrics_dict['score'] = final_score
        return metrics_dict
