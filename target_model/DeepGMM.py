import torch.nn as nn
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.optim.lr_scheduler
import os
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
import matplotlib.ticker as ticker
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

class Deep_GMM(nn.Module):
    def __init__(self, input_size, n_hiddenlayer, hidden_size, prediction_length, spatial_dim, n_component):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = n_hiddenlayer
        self.T = prediction_length
        self.M = spatial_dim
        self.D = prediction_length * spatial_dim
        self.K = n_component
        self.batchsize = spatial_dim
        self.lstm = nn.LSTM(input_size, hidden_size, n_hiddenlayer, batch_first=True, dropout=0)
        self.pi_layer1 = nn.Linear(hidden_size, self.K, bias=True)
        self.pi_layer2 = nn.Linear(spatial_dim, 1, bias=True)
        self.tem_head = nn.ModuleList([nn.Linear(hidden_size, prediction_length) for _ in range(self.K)])  # t_k
        self.spa_head = nn.ModuleList([nn.Linear(hidden_size, spatial_dim) for _ in range(self.K)])  # s_k
        self.diag_head = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(self.K)])  # d_k
        self.mean_head1 = nn.ModuleList([nn.Linear(hidden_size, self.D) for _ in range(self.K)])  # μ_k
        self.mean_head2 = nn.ModuleList([nn.Linear(spatial_dim, 1) for _ in range(self.K)])  # μ_k
        self.softplus = nn.Softplus()

    def _component_params(self, h, k):
        mu_k1 = self.mean_head1[k](h)
        mu_k1 = mu_k1.transpose(0, 1)
        mu_k2 = self.mean_head2[k](mu_k1).reshape(-1)
        t_k  = self.tem_head[k](h)
        s_k  = self.spa_head[k](h)
        d_k  = self.softplus(self.diag_head[k](h)) + 1e-4
        return mu_k2, t_k, s_k, d_k

    def _materialize_cov(self,  t, s, d):
        tem = torch.matmul(t.mT, t)
        spa = torch.matmul(s.mT, s)
        part1 = torch.kron(tem, spa)
        I_T = torch.eye(self.T)
        Dmat = torch.diag_embed(d.reshape(-1))
        part2 = torch.kron(I_T, Dmat)
        cov = part1 + part2 + 1e-6 * torch.eye(self.D)
        return cov

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        _, (h_n, c_n) = self.lstm(x, (h0, c0))
        h = h_n[-1]

        if self.K==1:
            pi = torch.tensor([1.0])
        else:
            pi1 = self.pi_layer1(h)  # (B, hidden) -> (B, K)
            pi1 = pi1.transpose(0, 1) # (B, K) -> (K, B)
            pi2 = self.pi_layer2(pi1).reshape(-1)  # (K, B) -> (K, 1) -> (K,)
            pi = F.softmax(pi2 , dim=-1)

        mus, covs = [], []
        for k in range(self.K):
            mu_k, t_k, s_k, d_k = self._component_params(h, k)
            cov_k = self._materialize_cov(t_k, s_k, d_k)
            mus.append(mu_k)
            covs.append(cov_k)
        return pi, mus, covs

    def loss(self, pi, mus, covs, target):
        log_probs = []
        for k in range(self.K):
            mvn = MultivariateNormal(mus[k], covs[k])
            log_probs.append(mvn.log_prob(target))
        log_probs = torch.stack(log_probs)
        log_mix = torch.logsumexp(torch.log(pi) + log_probs, dim=0)  # (B,)
        return -log_mix.mean()

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for encoder_length in [8]:  # 3, 4, 5, 6, 7, 8, 9, 10, 11
        for n_component in [2]:  # 1,2,3
            for n_hiddenlayer in [1]: # 1,2
                for hidden_size in [32]: # 16, 32, 64,128
                    print("encoder_length", encoder_length)
                    total_start_time = time.perf_counter()
                    input_size = 1
                    output_size = 1
                    batch_size = 21  # 30, 20
                    spatial_dim = 21 # 30, 20
                    num_epochs = 200
                    learning_rate = 0.001
                    fac = 0.5
                    pat = 5
                    stop = 10
                    prediction_length = 7 # 6, 8

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model =  Deep_GMM(input_size, n_hiddenlayer, hidden_size, prediction_length, spatial_dim, n_component).to(device)
                    n_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print("n_parameter", n_parameter)

                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
                            weights, mus, covs = model(x)
                            loss = model.loss(weights, mus, covs, y)
                            loss.backward()
                            optimizer.step()
                            epoch_train_loss += loss.item()
                            path = rf"trained_model\Deep_GMM_{encoder_length}_{n_component}_{n_hiddenlayer}_{hidden_size}_{epoch + 1}.pt"
                            torch.save(model.state_dict(), path)

                        epoch_train_loss /= len(train_batchxs)
                        train_losses.append(epoch_train_loss)

                        model.eval()
                        epoch_val_loss = 0
                        with torch.no_grad():
                            for val_id in range(len(val_batchxs)):
                                x = val_batchxs[val_id].to(torch.float32)
                                y = val_batchys[val_id].permute(1, 0, 2).to(torch.float32).reshape(-1)
                                weights, mus, covs = model(x)
                                loss = model.loss(weights, mus, covs, y)
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
                    with open(rf"trained_model\Deep_GMM_{encoder_length}_{n_component}_{n_hiddenlayer}_{hidden_size}_{epoch + 1}.json", "w") as f:
                        json.dump({"train_loss": train_losses, "val_loss": val_losses, "training_time": total_training_time, "n_parameter": n_parameter}, f)
