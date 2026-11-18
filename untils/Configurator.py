import os
import yaml
import re
import torch

class Config(object):
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.getparameters(config_dict)
        self.config_dict['device'] = torch.device("cuda:"+str(self.config_dict['gpu_id']) if torch.cuda.is_available() else "cpu")

    '''加载yaml'''
    def getparameters(self, config_dict):
        file_list = ['overall.yaml']
        file_list.append(config_dict['data']+'.yaml')
        for file in file_list:
            cur_dir = os.path.join(os.getcwd(), 'configs', file)
            assert os.path.isfile(cur_dir), f'Config File not found: {cur_dir}'
            with open(cur_dir, 'r', encoding='utf-8') as f:
                self.config_dict.update(yaml.load(f.read(), Loader=self._build_yaml_loader()))

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        return loader
    

    '''让Config对象支持字典的访问方式'''
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.config_dict[key] = value

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        if key in self.config_dict:
            return self.config_dict[key]
        else:
            return None

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.config_dict

    def __str__(self):  
        args_info = '\n'
        args_info += '\n'.join(["{}={}".format(arg, value) for arg, value in self.config_dict.items()])
        args_info += '\n\n'
        return args_info
