import torch.nn as nn
import torch.optim as optim
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
from torch.distributions import MultivariateNormal
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


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, X, A_norm):
        Xw = self.lin(X)
        H  = A_norm @ Xw
        return H

class GCN_GMM(nn.Module):
    """
      Σ_k = (t_k t_k^T) ⊗ (s_k s_k^T)  +  I_T ⊗ diag(d_k)  +  jitter * I_D
    """
    def __init__(self, input_size, hidden_size, num_layers, T, M, n_component, adj):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.T = T
        self.M = M
        self.D = T * M
        self.K = n_component
        self.batchsize = M

        A = adj.clone().float()
        I = torch.eye(M, device=A.device)
        A_tilde = A + I
        deg = A_tilde.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[~torch.isfinite(deg_inv_sqrt)] = 0.0
        A_norm = deg_inv_sqrt.unsqueeze(1) * A_tilde * deg_inv_sqrt.unsqueeze(0)
        self.register_buffer("A_norm", A_norm)
        self.gcn = GCNLayer(input_size, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0
        )

        self.pi_layer1 = nn.Linear(hidden_size, self.K, bias=True)
        self.pi_layer2 = nn.Linear(self.batchsize, 1, bias=True)

        self.tem_head  = nn.ModuleList([nn.Linear(hidden_size, T) for _ in range(self.K)])   # t_k
        self.spa_head  = nn.ModuleList([nn.Linear(hidden_size, M) for _ in range(self.K)])   # s_k
        self.diag_head = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(self.K)])   # d_k

        self.mean_head1 = nn.ModuleList([nn.Linear(hidden_size, self.D) for _ in range(self.K)])  # μ_k
        self.mean_head2 = nn.ModuleList([nn.Linear(self.batchsize, 1) for _ in range(self.K)])    # μ_k

        self.softplus = nn.Softplus()

    def _encode_gcn_lstm(self, x):

        B, S, Fin = x.shape  # x: (B, S, input_size)
        gcn_outs = []
        for t in range(S):
            X_t = x[:, t, :]                   # (B, Fin)
            H_t = self.gcn(X_t, self.A_norm)   # (B, hidden_size)
            H_t = F.relu(H_t)
            gcn_outs.append(H_t)

        H_seq = torch.stack(gcn_outs, dim=1)

        batch_size = B
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        _, (h_n, c_n) = self.lstm(H_seq, (h0, c0))
        h = h_n[-1]

        return h

    def _component_params(self, h, k):
        mu_k1 = self.mean_head1[k](h)
        mu_k1 = mu_k1.transpose(0, 1)
        mu_k2 = self.mean_head2[k](mu_k1).reshape(-1)

        # t_k, s_k, d_k
        t_k  = self.tem_head[k](h)
        s_k  = self.spa_head[k](h)
        d_k  = self.softplus(self.diag_head[k](h)) + 1e-4

        return mu_k2, t_k, s_k, d_k

    def _materialize_cov(self, t, s, d):
        # Spatiotemporal part
        tem = torch.matmul(t.mT, t)   # (T, B) @ (B, T) -> (T, T)
        spa = torch.matmul(s.mT, s)   # (M, B) @ (B, M) -> (M, M)
        part1 = torch.kron(tem, spa)  # (T*M, T*M) = (D, D)

        device = tem.device
        dtype  = tem.dtype
        I_T = torch.eye(self.T, device=device, dtype=dtype)

        d_flat = d.reshape(-1)        # (B,)
        Dmat = torch.diag_embed(d_flat)  # (B, B)
        part2 = torch.kron(I_T, Dmat)    # (T*B, T*B) = (D, D)

        cov = part1 + part2 + 1e-6 * torch.eye(self.D, device=device, dtype=dtype)
        return cov

    def forward(self, x):

        h = self._encode_gcn_lstm(x)  # hidden_size

        if self.K == 1:
            pi = torch.tensor([1.0], device=h.device, dtype=h.dtype)
        else:
            pi1 = self.pi_layer1(h)        # (B, hidden_size) -> (B, K)
            pi1 = pi1.transpose(0, 1)      # (K, B)
            pi2 = self.pi_layer2(pi1)      # (K, 1)
            pi2 = pi2.reshape(-1)          # (K,)
            pi  = F.softmax(pi2, dim=-1)   # (K,)

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
        log_probs = torch.stack(log_probs)          # (K,)
        log_mix = torch.logsumexp(torch.log(pi) + log_probs, dim=0)
        return -log_mix.mean()

def generate_chain_adjacency(n):
        A = np.zeros((n, n), dtype=int)
        for i in range(n):
            if i > 0:
                A[i, i - 1] = 1
            if i < n - 1:
                A[i, i + 1] = 1
        return A


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for encoder_length in [8]:
        for n_component in [2]:
            for n_hiddenlayer in [1]:
                for hidden_size in [32]:
                    print("encoder_length", encoder_length)
                    total_start_time = time.perf_counter()
                    input_size = 1
                    output_size = 1
                    batch_size = 21
                    spatial_dim = 21
                    num_epochs = 100
                    learning_rate = 0.001
                    fac = 0.5
                    pat = 3
                    stop = 6
                    prediction_length = 6
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    adj_matrix_np = generate_chain_adjacency(spatial_dim)
                    adj = torch.from_numpy(adj_matrix_np).to(device)

                    model = GCN_GMM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=n_hiddenlayer,
                        T=prediction_length,
                        M=spatial_dim,
                        n_component=n_component,
                        adj=adj
                    ).to(device)

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

                            path = rf"trained-model\GCN_GMM_{encoder_length}_{n_component}_{n_hiddenlayer}_{hidden_size}_{epoch + 1}.pt"
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
                    with open(rf"trained-model\GCN_GMM_{encoder_length}_{n_component}_{n_hiddenlayer}_{hidden_size}_{epoch + 1}.json", "w") as f:
                        json.dump({"train_loss": train_losses, "val_loss": val_losses, "training_time": total_training_time, "n_parameter": n_parameter}, f)

