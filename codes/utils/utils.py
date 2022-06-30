import re
import os
import yaml


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Namespace(object):
    def __init__(self, some_dict):
        for key, value in some_dict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            elif isinstance(value, list):
                value = value.copy()
                if isinstance(value[0], dict):
                    for i in range(len(value)):
                        value[i] = Namespace(value[i])
                self.__dict__[key] = value
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):

        raise AttributeError(
            f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")


def deep_update_dict(res_dict, in_dict):
    for key in in_dict.keys():
        if key in res_dict and isinstance(in_dict[key], dict) and isinstance(res_dict[key], dict) and \
                'name' in in_dict[key].keys() and 'kwargs' in in_dict[key].keys() and \
                'name' in res_dict[key].keys() and 'kwargs' in res_dict[key].keys() and \
                in_dict[key]['name'] == res_dict[key]['name']:
            deep_update_dict(res_dict[key]['kwargs'], in_dict[key]['kwargs'])
        else:
            res_dict[key] = in_dict[key]


def parse_yaml(yaml_file_path):
    res = {}
    with open(yaml_file_path, 'r') as yaml_file:
        f = yaml.load(yaml_file, Loader=yaml.FullLoader)
        if 'include' in f:
            abs_path = os.path.abspath(yaml_file_path)
            abs_path = abs_path.replace('\\', '/')
            abs_path_list = abs_path.split('/')
            for include_file_path in f['include']:
                include_file_path = include_file_path.replace('\\', '/')
                include_path_list = include_file_path.split('/')
                if '' in include_path_list:
                    include_path_list.remove('')
                if '.' in include_path_list:
                    include_path_list.remove('.')
                n = include_path_list.count('..')
                include_file_path = '/'.join(abs_path_list[:-(n + 1)] + include_path_list[n:])
                with open(include_file_path, 'r') as include_file:
                    deep_update_dict(res, yaml.load(include_file, Loader=yaml.FullLoader))
                    # res.update(yaml.load(include_file, Loader=yaml.FullLoader))
        deep_update_dict(res, f)
        # res.update(f)
        if 'include' in res.keys():
            res.pop('include')
    return res