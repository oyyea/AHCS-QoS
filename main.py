import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import argparse
from untils.Configurator import Config
from untils.logger import init_logger
from logging import getLogger
from untils.dataset import MyDataset
from itertools import product
import random
import numpy as np
import torch
from models.hyperinter import HyperInterModel
from models.hgnn_qos import HGNNQoSModel
from untils.trainer import Trainer
import datetime
import json
from torch.utils.data import DataLoader
import gc


def init_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_trial(filepath: str, trial_number: int, hyperparams: dict, metrics: dict):
    record = {
        "trial": trial_number,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "hyperparams": hyperparams,
        "metrics": metrics,
    }
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', '-g', type=int, default=0, help='gpu number')
    parser.add_argument('--data', '-d', type=str, default='rt', help='name of date')
    parser.add_argument('--model', '-m', choices=['ours', 'hgnn'], default='ours', help='model to train')

    args, _ = parser.parse_known_args()

    config_dict = {
        'gpu_id': args.gpu_id,
        'data': args.data,
        'model': args.model
    }

    config = Config(config_dict)

    init_logger(config)
    logger = getLogger()

    logger.info('dataset loaded……')
    dataset = MyDataset(config)
    logger.info('user number: {}, ws number: {}'.format(dataset.alluser_num, dataset.allws_num))

    # 网格搜索
    hyper_ls = []
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)

    idx = 0
    jsonfilename = f"{args.model}_" + datetime.datetime.now().strftime('%b-%d-%Y-%H-%M')+'.jsonl'
    jsonfilepath = os.path.join(os.getcwd(), 'log', jsonfilename)

    cut = 0
    mae_list = []
    rmse_list = []

    for hyper_tuple in combinators:
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])
        g = torch.Generator()
        g.manual_seed(config['seed'])
        g2 = torch.Generator()
        g2.manual_seed(config['seed'])
        g3 = torch.Generator()
        g3.manual_seed(config['seed'])
        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))
        logger.info(config)

        train_data, val_data, test_date, cold_user, cold_ws = dataset.split()
        logger.info(f"valid user num: {train_data.getusernum()}, valid ws num: {train_data.getwsnum()}")

        if args.model == 'hgnn' or (args.model == 'ours' and config['hgnn'] == 'hgnn'):
            use_hg, ws_hg = train_data.build_hypergraph_all()
            use_hg = use_hg.to(config['device'])
            ws_hg = ws_hg.to(config['device'])
        else:
            use_hg, ws_hg = None, None

        train_workers = config['num_work']
        eval_workers = config['num_work']//2
        train_persistent = train_workers > 0
        eval_persistent = eval_workers > 0
        pin = config['device'].type != 'cpu'
        train_loader = DataLoader(train_data, batch_size=config['train_batch_size'], shuffle=True, num_workers=train_workers, pin_memory=pin,
                                  persistent_workers=train_persistent, worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_data, batch_size=config['eval_batch_size'], shuffle=False, num_workers=eval_workers, pin_memory=pin,
                                persistent_workers=eval_persistent, worker_init_fn=seed_worker, generator=g2)
        test_loader = DataLoader(test_date, batch_size=config['eval_batch_size'], shuffle=False, num_workers=eval_workers, pin_memory=pin,
                                 persistent_workers=eval_persistent, worker_init_fn=seed_worker, generator=g3)

        if args.model == 'ours':
            model = HyperInterModel(config, train_data.getusernum(), train_data.getwsnum(), use_hg, ws_hg)
        else:
            config['ubeta'] = 0.0
            config['wbeta'] = 0.0
            config['dbeta'] = 0.0
            model = HGNNQoSModel(config, train_data.getusernum(), train_data.getwsnum(), use_hg, ws_hg)
        model = model.to(config['device'])
        logger.info(f"\n总参数量：{sum(p.numel() for p in model.parameters()):,}\n")

        trainer = Trainer(config, model)
        val_mae, test_mae, test_rmse = trainer.fit(train_loader, val_loader, test_loader, cold_user=cold_user, cold_ws=cold_ws)

        idx += 1

        save_trial(jsonfilepath, idx, {k: config[k] for k in config['hyper_parameters']}, {"val_mae": val_mae, "test_mae": test_mae, "test_rmse": test_rmse})

        if config['final_test']:
            cut += 1
            mae_list.append(test_mae)
            rmse_list.append(test_rmse)
            if cut == 5:
                mean_mae = float(np.mean(mae_list))
                std_mae = float(np.std(mae_list))
                mean_rmse = float(np.mean(rmse_list))
                std_rmse = float(np.std(rmse_list))
                logger.info(
                    f"\n  * Density={config['train_ratio'] * 100:.1f}%  "
                    f"MAE={mean_mae:.4f}+-{std_mae:.4f}  "
                    f"RMSE={mean_rmse:.4f}+-{std_rmse:.4f}"
                )
                mae_list.clear()
                rmse_list.clear()
                cut = 0

        # del model, trainer
        # del train_loader, val_loader, test_loader
        # del use_hg, ws_hg, intergraph
        # gc.collect()
        # torch.cuda.empty_cache()





        

    


