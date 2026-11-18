import pandas as pd
import numpy as np
from logging import getLogger
import os
from dhg import Hypergraph, BiGraph
import torch
from torch.utils.data import Dataset
from sklearn.cluster import DBSCAN
from collections import defaultdict
# import h3

class MyDataset(Dataset):
    def __init__(self, config, all_data = None):
        self.config = config
        self.logger = getLogger()
        self.alluser_num = 0
        self.allws_num = 0
        base_dir = os.getcwd()
        self.data_path = os.path.join(base_dir, 'data', self.config['data_path'])
        if all_data is None:
            target_data = self.load_interaction(os.path.join(self.data_path, self.config['target_data']))
            rows, cols = np.where(target_data > 0)
            mask = ~np.isin(cols, [4700, 4701]) # wslist里未存在该编号
            rows, cols = rows[mask], cols[mask]
            ratings = target_data[rows, cols]
            self.all_data = np.stack([rows, cols, ratings], axis=1)
            self.alluser_num = target_data.shape[0]
            self.allws_num = target_data.shape[1]
        else:
            self.all_data = all_data

    def load_attr(self, filepath: str, sep='\t') -> pd.DataFrame:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f'File not found: {filepath}')
        df = pd.read_csv(
            filepath,
            sep=sep,
            skiprows=[1],
            encoding="ISO-8859-1"
        )
        df.columns = df.columns.str.replace(r'[\[\]]', '', regex=True)
        df["AS"] = df["AS"].str.split().str[0]
        return df   

    def load_interaction(self, filepath: str):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f'File not found: {filepath}')
        return np.loadtxt(filepath, dtype=np.float32)
    
    def split(self):
        test_cold_user=np.array([], dtype=int)
        test_cold_ws=np.array([], dtype=int)
        filter_user = self.config['filter_out_cod_start_users']
        filter_ws = self.config['filter_out_cod_start_wss']
        rng = np.random.default_rng(self.config['seed'])
        if filter_user or filter_ws:
            test_mask = np.zeros(len(self.all_data), dtype=bool)
            if filter_user:
                test_cold_user = rng.choice(self.alluser_num, 169, replace=False)
                test_mask |= np.isin(self.all_data[:, 0], test_cold_user)

            if filter_ws:
                test_cold_ws = rng.choice(self.allws_num, 300, replace=False)
                test_mask |= np.isin(self.all_data[:, 1], test_cold_ws)

            test_data = self.all_data[test_mask]

            train_candidate_data = self.all_data[~test_mask]
            perm = rng.permutation(len(train_candidate_data))
            n_train = max(1, int(len(train_candidate_data) * self.config['train_ratio'] * 0.9))
            n_val = max(1, int(len(train_candidate_data) * self.config['train_ratio'] * 0.1))
            train_data = train_candidate_data[perm[:n_train]]
            val_data = train_candidate_data[perm[n_train:n_train + n_val]]
            test_data = np.concatenate((test_data, train_candidate_data[perm[n_train+n_val:]]), axis=0)
        else:
            perm = rng.permutation(len(self.all_data))
            n_train = max(1, int(len(self.all_data) * self.config['train_ratio'] * 0.9))
            n_val = max(1, int(len(self.all_data) * self.config['train_ratio'] * 0.1))

            train_data = self.all_data[perm[:n_train]]
            val_data = self.all_data[perm[n_train:n_train+n_val]]
            test_data = self.all_data[perm[n_train+n_val:]]
        dfs = [train_data, val_data, test_data]
        return *[self.copy(_) for _ in dfs],test_cold_user,test_cold_ws
    
    def copy(self, data):
        spdataset = MyDataset(self.config, all_data=data)
        spdataset.alluser_num = self.alluser_num
        spdataset.allws_num = self.allws_num
        return spdataset

    def build_hypergraph_all(self):
        user_attr = self.load_attr(os.path.join(self.data_path, self.config['user_data']))
        ws_attr = self.load_attr(os.path.join(self.data_path, self.config['ws_data']))
        user_hg = self.build_hypergraph(user_attr, is_ws=False)
        ws_hg = self.build_hypergraph(ws_attr, is_ws=True)
        return user_hg, ws_hg

    def build_hypergraph(self, df: pd.DataFrame, is_ws) -> Hypergraph:
        df.index = range(len(df))

        if is_ws:
            edges_list = self.config['ws_attr']
            strout="WS hypergraph:"
        else:
            edges_list = self.config['user_attr']
            strout="User hypergraph:"
        
        all_edges = []
        for attr in edges_list:
            strout+=f"{attr} edges: "
            if isinstance(attr, list):
                missing = [col for col in attr if col not in df.columns]
                if missing:
                    raise ValueError(f"Columns not found: {missing}")
                valid_mask = df[attr].notna().all(axis=1)
                valid_indices = df.index[valid_mask].tolist()
                if self.config['up_model'] == 'knn':
                    coords = torch.tensor(
                        df.loc[valid_mask, attr].values,
                        dtype=torch.float32
                    )
                    k = self.config['k_i'] if is_ws else self.config["k_u"]
                    location_hg_tmp = Hypergraph.from_feature_kNN(coords, k=k)
                    edges = [
                        [valid_indices[n] for n in edge]
                        for edge in location_hg_tmp.e[0]
                    ]
                elif self.config['up_model'] == 'dbscan':
                    coords_np = df.loc[valid_mask, attr].values
                    coords_rad = np.radians(coords_np)
                    if is_ws:
                        eps = self.config['i_eps']
                        min_samples = self.config['i_minpts']
                    else:
                        eps = self.config['u_eps']
                        min_samples = self.config['u_minpts']
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples,
                                    metric='haversine', algorithm='ball_tree')
                    labels = dbscan.fit_predict(coords_rad)
                    clusters = defaultdict(list)
                    for local_idx, label in enumerate(labels):
                        if label != -1:
                            clusters[label].append(local_idx)
                    total_nodes = len(labels)
                    noise_nodes = np.count_nonzero(labels == -1)
                    noise_ratio = noise_nodes / total_nodes
                    self.logger.info(f"孤岛节点比例: {noise_ratio:.2%}")
                    edges = [
                        [valid_indices[n] for n in cluster_nodes]
                        for cluster_nodes in clusters.values()
                    ]
                # elif self.config['up_model'] == 'h3':
                #     resolution = self.config['i_h3'] if is_ws else self.config['u_h3']
                #     valid_df = df.loc[valid_mask]
                #     h3_to_nodes = defaultdict(list)
                #     lats = valid_df[attr[0]].values
                #     lngs = valid_df[attr[1]].values
                #     for node_idx, lat, lng in zip(valid_indices, lats, lngs):
                #         h3_index = h3.latlng_to_cell(lat, lng, resolution)
                #         h3_to_nodes[h3_index].append(node_idx)
                #     edges = [nodes for nodes in h3_to_nodes.values() if len(nodes) > 1]
            else:
                if attr not in df.columns:
                    raise ValueError(f"Attribute '{attr}' not found in DataFrame columns.")
                edges = [list(v) for v in df.groupby(attr).groups.values() if len(v) > 1]
            strout += f"{len(edges)}\t"
            all_edges.extend(edges)
        self.logger.info(f"{strout}")
        nodenum = self.allws_num if is_ws else self.alluser_num
        return Hypergraph(nodenum, all_edges)

    def getusernum(self):
        return self.alluser_num

    def getwsnum(self):
        return self.allws_num
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        u, i, r = self.all_data[idx]
        return int(u), int(i), float(r)
    

    

        




            
        

