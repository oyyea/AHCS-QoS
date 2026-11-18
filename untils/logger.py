import logging
import os
import datetime

def init_logger(config):
    base_dir = os.getcwd()
    logpath = os.path.join(base_dir, 'log')
    os.makedirs(logpath, exist_ok=True)
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y-%H-%M')
    if config['model'] == 'ours':
        model_name = '{}'.format(config['up_model'])
    else:
        model_name = config['model']
    logfilename = '{}-{}-{}.log'.format(model_name, config['data'], cur)
    logfilepath = os.path.join(logpath, logfilename)

    fmt = "%(asctime)-15s %(message)s" 
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    level = logging.INFO

    #文件输出
    fh = logging.FileHandler(logfilepath, 'w', 'utf-8')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    #控制台输出
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)

    logging.basicConfig(
        level=level,
        handlers=[sh, fh]
    )
