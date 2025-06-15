import os
import sys
import logging


class Logger:
    def __init__(self, cfg):
        logdir = os.path.join(str(cfg.logger.dir), str(cfg.model.name), cfg.task_start_time)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        cfg.logger.path = logdir

        log_file_name = '{}/logs.txt'.format(logdir)
        log_format = '%(asctime)s - %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)

        self.fh = logging.FileHandler(log_file_name, mode='a+')
        self.fh.setFormatter(logging.Formatter(log_format))

        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(self.fh)

    def log_msg(self, message):
        self.logger.info(message)

    def log_cfg(self, cfg):
        cfg_dict = cfg.get_cfg_dict()
        if hasattr(cfg_dict, '__dict__'):
            cfg_dict = cfg_dict.__dict__
        key_max_length = max(list(map(len, cfg_dict.keys())))

        cfg_str = ''
        temp_k = ''
        for k, v in cfg_dict.items():
            if k.split('|')[0] != temp_k:
                temp_k = k.split('|')[0]
                cfg_str += '-------------------------------------------------\n'
            cfg_str += ('{'+':<{}'.format(key_max_length)+'} : {'+':<{}'.format(key_max_length)+'}').format(k, str(v))
            cfg_str += '\n'
        cfg_str = cfg_str.strip()

        self.logger.info(f'- - - > Logging configs\n{cfg_str}\n-------------------------------------------------\n')


def get_logger(cfg):
    return Logger(cfg)
