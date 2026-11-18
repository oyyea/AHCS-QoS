from logging import getLogger
import os
import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np

class SHuberLoss(nn.Module):
    def __init__(self, delta=0.5, psi=0.05):
        super(SHuberLoss, self).__init__()
        self.delta = delta
        self.psi = psi
    def forward(self, pred, target):
        error = torch.abs(pred - target)
        loss = torch.where(
            error < self.delta,
            0.5 * (error ** 2),
            self.psi * (self.delta * error - 0.5 * (self.delta ** 2))
        )
        return loss.mean()

class Trainer(object):
    def __init__(self, config, model):
        self.model = model
        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.epochs = config['epochs']
        self.patience = config['patience']
        self.device = config['device']
        self.optimizer = self._build_optimizer(lr=self.learning_rate)
        self.lossf = config['lossf']
        self.colduser = config['filter_out_cod_start_users']
        self.coldws = config['filter_out_cod_start_wss']
        self.ubeta = config['ubeta']
        self.wbeta = config['wbeta']
        self.dbeta = config['dbeta']
        self.eps = config['eps']
        self.loss_fnc = self.get_losser()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=config["factor"], patience=config["lrpatience"], min_lr=1e-5)
        self.data_name = config['data']
        self.model_name = config['model']
        self.seed = config['seed']
        self.train_ratio = config['train_ratio']


    def get_losser(self):
        if self.lossf == 'L1':
            return nn.L1Loss()
        elif self.lossf == 'SL1':
            return SHuberLoss(delta=self.eps)

    def _build_optimizer(self, lr=None):
        if self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.learner.lower() == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        return optimizer
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for u, i, r in train_loader:
            u, i, r = u.to(self.device), i.to(self.device), r.to(self.device)
            self.optimizer.zero_grad()
            pred, pred2, pred3, pred4 = self.model(u, i)
            loss = self.loss_fnc(pred, r)+self.dbeta * self.loss_fnc(pred2, r) + self.wbeta * self.loss_fnc(pred3, r) + self.ubeta * self.loss_fnc(pred4, r)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(r)
        avg_loss = total_loss / len(train_loader.dataset)
        return avg_loss

    def fit(self, train_loader, val_loader, test_loader, cold_user=None, cold_ws=None):
        best_mae = float('inf')
        pat = 0
        total_start = time.time()
        for epoch in range(1, self.epochs):
            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader)
            mae, rmse = self.evaluate(val_loader)
            self.scheduler.step(mae)
            # self.scheduler.step()
            epoch_time = time.time() - epoch_start

            tag = ""
            if mae < best_mae - 1e-5:
                best_mae = mae
                torch.save(self.model.state_dict(), "./log/best_model.pt")
                tag = "  ← best"
                pat = 0
            else:
                pat += 1
                if pat > self.patience:
                    break

            self.logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"{epoch_time:.1f}s | "
                f"Train L1={train_loss:.4f} | "
                f"Val MAE={mae:.4f}  RMSE={rmse:.4f}"
                f"{tag}"
            )

        self.model.load_state_dict(torch.load('./log/best_model.pt', map_location=self.device))
        test_mae, test_rmse = self.evaluate(test_loader, is_test=True, cold_user=cold_user, cold_ws=cold_ws)
        total_time = time.time() - total_start
        self.logger.info(f"Total training time: {total_time:.2f}s, Best val MAE={best_mae:.4f}, Best Test MAE={test_mae:.4f}  RMSE={test_rmse:.4f}")
        return best_mae, test_mae, test_rmse


    @torch.no_grad()
    def evaluate(self, eval_loader, is_test=False, cold_user=None, cold_ws=None):
        self.model.eval()
        preds, targets, users, items = [], [], [], []
        for u, i, r in eval_loader:
            u, i, r = u.to(self.device), i.to(self.device), r.to(self.device)
            pred, pred2, pred3, pred4 = self.model(u, i)
            preds.append(pred.detach().cpu())
            targets.append(r.cpu())
            if is_test:
                users.append(u.cpu())
                items.append(i.cpu())
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        mae = (preds - targets).abs().mean().item()
        rmse = (preds - targets).pow(2).mean().sqrt().item()

        if is_test and (self.colduser or self.coldws):
            test_u_np = torch.cat(users).numpy()
            test_s_np = torch.cat(items).numpy()
            is_cold_u = np.isin(test_u_np, cold_user)
            is_cold_s = np.isin(test_s_np, cold_ws)
            mask_dc = torch.from_numpy(is_cold_u & is_cold_s)
            mask_uc = torch.from_numpy(is_cold_u & ~is_cold_s)
            mask_sc = torch.from_numpy(~is_cold_u & is_cold_s)
            mask_normal = torch.from_numpy(~is_cold_u & ~is_cold_s)

            def get_mae_safe(mask_tensor):
                if mask_tensor.any():
                    return (preds[mask_tensor] - targets[mask_tensor]).abs().mean().item()
                return 0.0

            def get_rmse_safe(mask_tensor):
                if mask_tensor.any():
                    squared_error = (preds[mask_tensor] - targets[mask_tensor]).pow(2)
                    return squared_error.mean().sqrt().item()
                return 0.0

            mae_dc = get_mae_safe(mask_dc)
            mae_uc = get_mae_safe(mask_uc)
            mae_sc = get_mae_safe(mask_sc)
            mae_normal = get_mae_safe(mask_normal)
            rmse_dc = get_rmse_safe(mask_dc)
            rmse_uc = get_rmse_safe(mask_uc)
            rmse_sc = get_rmse_safe(mask_sc)
            rmse_normal = get_rmse_safe(mask_normal)

            self.logger.info(f"  ┌─ Normal MAE: {mae_normal:.4f}")
            self.logger.info(f"  ├─ UC MAE: {mae_uc:.4f}")
            self.logger.info(f"  ├─ SC MAE: {mae_sc:.4f}")
            self.logger.info(f"  └─ DC MAE: {mae_dc:.4f}")
            self.logger.info(f"  ┌─ Normal RMSE: {rmse_normal:.4f}")
            self.logger.info(f"  ├─ UC RMSE: {rmse_uc:.4f}")
            self.logger.info(f"  ├─ SC RMSE: {rmse_sc:.4f}")
            self.logger.info(f"  └─ DC RMSE: {rmse_dc:.4f}")

            # output_dir = os.path.join(os.getcwd(), 'output', 'cold_start_data')
            # os.makedirs(output_dir, exist_ok=True)
            # density = f"{float(self.train_ratio) * 100:g}".replace('.', 'p')
            # output_path = os.path.join(
            #     output_dir,
            #     f"{self.data_name}_{self.model_name}_density_{density}_seed_{self.seed}.npz",
            # )
            # np.savez_compressed(
            #     output_path,
            #     user_id=test_u_np.astype(np.int64, copy=False),
            #     service_id=test_s_np.astype(np.int64, copy=False),
            #     target=targets.numpy().astype(np.float32, copy=False),
            #     prediction=preds.numpy().astype(np.float32, copy=False),
            #     cold_user_ids=np.asarray(cold_user, dtype=np.int64),
            #     cold_service_ids=np.asarray(cold_ws, dtype=np.int64),
            #     is_cold_user=is_cold_u,
            #     is_cold_service=is_cold_s,
            # )
            # self.logger.info(f"Cold-start test data saved to: {output_path}")
        # if is_test:
        #     u_z,w_z,u_h2,w_h2,u_hg, w_hg, u_lg, w_lg,user_gate,ws_gate = self.model.extract_embeddings()
        #     u_z = u_z.cpu().numpy()
        #     w_z = w_z.cpu().numpy()
        #     u_h2 = u_h2.cpu().numpy()
        #     w_h2 = w_h2.cpu().numpy()
        #     u_hg = u_hg.cpu().numpy()
        #     w_hg = w_hg.cpu().numpy()
        #     u_lg = u_lg.cpu().numpy()
        #     w_lg = w_lg.cpu().numpy()
        #     u_w = user_gate.cpu().numpy()
        #     w_w = ws_gate.cpu().numpy()
        #     np.savez('full_nocold.npz',u_z=u_z, w_z=w_z, u_h2=u_h2, w_h2=w_h2, u_hg=u_hg, w_hg=w_hg, u_lg=u_lg, w_lg=w_lg,u_w=u_w,w_w=w_w)
        return mae, rmse


