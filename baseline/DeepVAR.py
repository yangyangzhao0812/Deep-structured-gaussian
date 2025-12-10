import torch.nn as nn
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.optim.lr_scheduler
import os
# os.chdir("../../..")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from feature_engine.creation import CyclicalFeatures
import seaborn as sns
import random
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import time
import json
from matplotlib.ticker import FormatStrFormatter
plt.rcParams["font.family"] = "Times New Roman"


with open(r"train_batchx.pickle", "rb") as f:
    train_batchxs = pickle.load(f)
with open(r"train_batchy.pickle", "rb") as f:
    train_batchys = pickle.load(f)

with open(r"val_batchx.pickle", "rb") as f:
    val_batchxs = pickle.load(f)
with open(r"val_batchy.pickle", "rb") as f:
    val_batchys = pickle.load(f)

with open(r"test_batchx.pickle", "rb") as f:
    test_batchxs = pickle.load(f)
with open(r"test_batchy.pickle", "rb") as f:
    test_batchys = pickle.load(f)

print("train_batchxs, val_batchxs, test_batchxs", f"{len(train_batchxs), len(val_batchxs), len(test_batchxs)}")


class DeepVAR_GMM(nn.Module):
    """
        Σ_k = L_k L_k^T + diag(d_k) + jitter * I_D,
    """
    def __init__(self, input_size, num_layers, hidden_size,
                 T, M, low_rank, n_component, jitter=1e-6, batchsize=30):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.T = T
        self.M = M
        self.D = T * M
        self.R = low_rank
        self.K = n_component
        self.batchsize = batchsize
        self.jitter = jitter

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.0,
        )

        self.pi_layer1 = nn.Linear(hidden_size, self.K, bias=True)      # (B, H) -> (B, K)
        self.pi_layer2 = nn.Linear(self.batchsize, 1, bias=True)        # (K, B) -> (K, 1)

        self.mean_head1 = nn.ModuleList([nn.Linear(hidden_size, self.D) for _ in range(self.K)])  # (B, H) -> (B, D)
        self.mean_head2 = nn.ModuleList([nn.Linear(self.batchsize, 1) for _ in range(self.K)])  # (D, B) -> (D, 1) -> (D,)

        self.factor_head1 = nn.ModuleList([nn.Linear(hidden_size, self.D) for _ in range(self.K)])  # (B, H) -> (B, D)
        self.factor_head2 = nn.ModuleList([nn.Linear(self.batchsize, self.R) for _ in range(self.K)])

        self.diag_head1 = nn.ModuleList([nn.Linear(hidden_size, self.D) for _ in range(self.K)])  # (B, H) -> (B, D)
        self.diag_head2 = nn.ModuleList([nn.Linear(self.batchsize, 1) for _ in range(self.K)])  # (D, B) -> (D, 1) -> (D,)

        self.softplus = nn.Softplus()

    def _component_params(self, h, k):

        mu_k1 = self.mean_head1[k](h)       # (B, D)
        mu_k1 = mu_k1.transpose(0, 1)       # (D, B)
        mu_k2 = self.mean_head2[k](mu_k1)   # (D, 1)
        mu_k = mu_k2.reshape(-1)            # (D,)

        fac1 = self.factor_head1[k](h)      # (B, D)
        fac1 = fac1.transpose(0, 1)         # (D, B)
        L_k  = self.factor_head2[k](fac1)   # (D, R)

        d1 = self.diag_head1[k](h)          # (B, D)
        d1 = d1.transpose(0, 1)             # (D, B)
        d2 = self.diag_head2[k](d1)         # (D, 1)
        d_k = self.softplus(d2.reshape(-1)) + 1e-4   # (D,) positive

        return mu_k, L_k, d_k

    def _materialize_cov(self, L, d):
        cov_lowrank = L @ L.T                # (D, D)
        cov_diag = torch.diag(d)             # (D, D)
        I_D = torch.eye(self.D, device=L.device, dtype=L.dtype)
        cov = cov_lowrank + cov_diag + self.jitter * I_D
        return cov

    def forward(self, x):
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        _, (h_n, c_n) = self.lstm(x, (h0, c0))
        h = h_n[-1]   # (B, H)

        if self.K == 1:
            pi = torch.tensor([1.0], device=x.device, dtype=h.dtype)
        else:
            pi1 = self.pi_layer1(h)              # (B, K)
            pi1 = pi1.transpose(0, 1)            # (K, B)
            pi2 = self.pi_layer2(pi1).reshape(-1) # (K,)
            pi = F.softmax(pi2, dim=-1)          # (K,)

        mus, covs = [], []
        for k in range(self.K):
            mu_k, L_k, d_k = self._component_params(h, k)
            cov_k = self._materialize_cov(L_k, d_k)
            mus.append(mu_k)
            covs.append(cov_k)

        return pi, mus, covs

    def loss(self, pi, mus, covs, target):
        log_probs = []
        for k in range(self.K):
            mvn = MultivariateNormal(mus[k], covs[k])
            log_probs.append(mvn.log_prob(target))

        log_probs = torch.stack(log_probs)
        log_mix = torch.logsumexp(torch.log(pi) + log_probs, dim=0)
        return -log_mix.mean()


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    total_start_time = time.perf_counter()
    for encoder_length in [8]:  # 3, 4, 5, 6, 7, 8, 9, 10,
        for n_component in [2]:  # 1,2,3
            for n_hiddenlayer in [1]: # 1,2
                for hidden_size in [32]: # 16, 32, 64, 128
                    print("encoder_length", encoder_length)
                    prediction_length = 8
                    input_size = 1
                    output_size = 1
                    batch_size = 20
                    num_epochs = 200
                    spatial_dim = batch_size
                    learning_rate = 0.001
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    model = DeepVAR_GMM(
                        input_size=1,
                        num_layers=1,
                        hidden_size=hidden_size,
                        T=prediction_length,
                        M=batch_size,
                        low_rank=10,
                        n_component=2,
                        jitter=1e-6,
                        batchsize=batch_size)

                    n_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print("n_parameter", n_parameter)

                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    fac = 0.5
                    pat = 5
                    stop = 10
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=fac, patience=pat)
                    early_stopping_patience = stop
                    best_loss = float('inf')
                    epochs_no_improve = 0
                    train_losses = []
                    val_losses = []
                    for epoch in range(num_epochs):
                        model.train()
                        epoch_train_loss = 0.0
                        print("epoch", "\t", epoch)
                        print("lr", "\t", scheduler.get_last_lr())
                        for train_id in range(len(train_batchxs)):
                            x = train_batchxs[train_id].to(torch.float32)
                            y = train_batchys[train_id].permute(1, 0, 2).to(torch.float32).reshape(-1)
                            optimizer.zero_grad()
                            pi, mean, cov = model(x)
                            loss = model.loss(pi, mean, cov, y)
                            loss.backward()
                            optimizer.step()
                            epoch_train_loss += loss.item()
                            path = rf"trained-model\DeepVAR_{encoder_length}_{n_component}_{n_hiddenlayer}_{hidden_size}_{epoch + 1}.pt"
                            torch.save(model.state_dict(), path)

                        epoch_train_loss /= len(train_batchxs)
                        train_losses.append(epoch_train_loss)

                        model.eval()
                        epoch_val_loss = 0
                        with torch.no_grad():
                            for val_id in range(len(val_batchxs)):
                                x = val_batchxs[val_id].to(torch.float32)
                                y = val_batchys[val_id].permute(1, 0, 2).to(torch.float32).reshape(-1)
                                pi, mus, covs = model(x)
                                loss = model.loss(pi, mus, covs, y)
                                epoch_val_loss += loss.item()
                            epoch_val_loss = epoch_val_loss / len(val_batchxs)
                            val_losses.append(epoch_val_loss)
                            if epoch_val_loss < best_loss:
                                best_loss = epoch_val_loss
                                epochs_no_improve = 0
                            else:
                                epochs_no_improve += 1
                            print(
                                f"Epoch {epoch + 1}: Val Loss = {epoch_val_loss:.4f}, Patience = {epochs_no_improve}")
                            if epochs_no_improve >= early_stopping_patience:
                                print("Early stopping triggered.")
                                break
                            if (epoch + 1) % 10 == 0:
                                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
                        scheduler.step(epoch_val_loss)
                    total_training_time = time.perf_counter() - total_start_time
                    with open(rf"trained-model\DeepVAR_{encoder_length}_{n_component}_{n_hiddenlayer}_{hidden_size}_{epoch + 1}.json", "w") as f:
                        json.dump({"train_loss": train_losses, "val_loss": val_losses, "training_time": total_training_time, "n_parameter": n_parameter}, f)
