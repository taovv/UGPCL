import os
import logging

from torch import Tensor
from tensorboardX import SummaryWriter


class Logger(object):
    def __init__(self,
                 log_dir,
                 logger_name='root',
                 log_level=logging.INFO,
                 log_file=True,
                 file_mode='w',
                 tensorboard=True):

        assert log_dir is not None, 'log_dir is None!'

        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=f'{self.log_dir}/tensorboard_log') if tensorboard else None
        self.logger = self.init_logger(logger_name, log_file, log_level, file_mode)

    def init_logger(self, name, log_file, log_level, file_mode):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        stream_handler = logging.StreamHandler()
        handlers = [stream_handler]
        if log_file:
            log_file = os.path.join(self.log_dir, 'info.log')
            file_handler = logging.FileHandler(log_file, file_mode)
            handlers.append(file_handler)

        date_format = '%Y-%m-%d %H:%M:%S'
        # basic_format = '%(asctime)s-%(name)s-%(levelname)s-%(message)s'
        basic_format = '%(asctime)s - %(name)s: %(message)s'
        formatter = logging.Formatter(basic_format, date_format)
        for handler in handlers:
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
            logger.addHandler(handler)

        logger.setLevel(log_level)
        return logger

    def update_scalars(self, ordered_dict, step):
        for key, value in ordered_dict.items():
            if isinstance(value, Tensor):
                ordered_dict[key] = value.item()
            self.writer.add_scalar(key, value, step)

    def update_images(self, images_dict, step):
        for key, value in images_dict.items():
            self.writer.add_image(key, value, step)

    def info(self, *kwargs):
        self.logger.info(*kwargs)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        self.logger.handlers.clear()
        del self.logger
