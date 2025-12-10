import torch.nn as nn
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.metrics import root_mean_squared_error
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
import seaborn as sns
import math
import random
from scipy.stats import norm
import torch.nn.functional as F
from scipy.optimize import bisect
import time
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
plt.rcParams["font.family"] = "Times New Roman"


def loaddata():
    route = 2  #1, 3
    data = pd.read_csv(r"Jan2Mar_TT_normal" + str(route) + ".csv", index_col=0)
    data["ACTTT"] = data["avl_segment_travel_time"] + data["avl_time_on_stop_at_fromsan"]
    train_val_data = data[data["date"] < "2022-03-23"].reset_index(drop=True)
    test_data = data[data["date"] >= "2022-03-23"].reset_index(drop=True)
    mu, sigma = torch.tensor(train_val_data.groupby('y')['ACTTT'].mean().to_numpy()), torch.tensor((train_val_data.groupby('y')['ACTTT'].std().to_numpy()))
    print("mu", mu)
    print("sigma", sigma)
    return mu, sigma


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

class Transformer_GMM_(nn.Module):
    """
      Σ_k = (t_k t_k^T) ⊗ (s_k s_k^T)  +  I_T ⊗ diag(d_k)  +  jitter * I_D
    """
    def __init__(self, input_size, n_hiddenlayer, hidden_size, T, M, n_component,
                 nhead, max_len):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = n_hiddenlayer
        self.T = T
        self.M = M
        self.D = T * M
        self.K = n_component

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_len, hidden_size)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=4 * hidden_size,
            dropout=0.0,
            batch_first=True,
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_hiddenlayer
        )

        self.pi_layer1 = nn.Linear(hidden_size, self.K, bias=True)
        self.pi_layer2 = nn.Linear(M, 1, bias=True)

        self.tem_head = nn.ModuleList([nn.Linear(hidden_size, T) for _ in range(self.K)])  # t_k: (B, T)
        self.spa_head = nn.ModuleList([nn.Linear(hidden_size, M) for _ in range(self.K)])  # s_k: (B, M)
        self.diag_head = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(self.K)])  # d_k: (B, 1)

        self.mean_head1 = nn.ModuleList([nn.Linear(hidden_size, self.D) for _ in range(self.K)])  # μ_k: (B, D)
        self.mean_head2 = nn.ModuleList([nn.Linear(M, 1) for _ in range(self.K)])  # μ_k: (D, 1)

        self.softplus = nn.Softplus()

    def _component_params(self, h, k):
        mu_k1 = self.mean_head1[k](h)       # (B, D)
        mu_k1 = mu_k1.transpose(0, 1)       # (D, B)
        mu_k2 = self.mean_head2[k](mu_k1)   # (D, 1)
        mu_k2 = mu_k2.reshape(-1)           # (D,)

        t_k = self.tem_head[k](h)           # (B, T)
        s_k = self.spa_head[k](h)           # (B, M)
        d_k = self.softplus(self.diag_head[k](h)) + 1e-4   # (B, 1)

        return mu_k2, t_k, s_k, d_k

    def _materialize_cov(self, t, s, d):
        # Spatiotemporal part
        tem = torch.matmul(t.mT, t)   # (T, B) @ (B, T) -> (T, T)
        spa = torch.matmul(s.mT, s)   # (M, B) @ (B, M) -> (M, M)
        part1 = torch.kron(tem, spa)  # (T*M, T*M) = (D, D)

        I_T = torch.eye(self.T, device=t.device, dtype=t.dtype)

        d_flat = d.reshape(-1)
        Dmat = torch.diag_embed(d_flat)  # (B, B)
        part2 = torch.kron(I_T, Dmat)    # (T*B, T*B) = (D, D)

        cov = part1 + part2 + 1e-6 * torch.eye(self.D, device=t.device, dtype=t.dtype)
        return cov

    def forward(self, x):
        B, S, _ = x.shape

        z = self.input_proj(x)  # (B, S, H)

        z = z + self.pos_embedding[:, :S, :]

        h_seq = self.transformer(z)  # (B, S, H)
        h = h_seq[:, -1, :]          # (B, H)

        if self.K == 1:
            pi = torch.tensor([1.0], device=h.device, dtype=h.dtype)
        else:
            pi1 = self.pi_layer1(h)       # (B, H) -> (B, K)
            pi1 = pi1.transpose(0, 1)     # (K, B)
            pi2 = self.pi_layer2(pi1)     # (K, 1)
            pi2 = pi2.reshape(-1)         # (K,)
            pi  = F.softmax(pi2, dim=-1)  # (K,)

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
        log_probs = torch.stack(log_probs)                 # (K,)
        log_mix = torch.logsumexp(torch.log(pi) + log_probs, dim=0)
        return -log_mix.mean()

def forecasting(mu, sigma, modeltype, n_component):

    start_time = time.time()
    mu = np.tile(mu, prediction_length)
    sigma = np.tile(sigma, prediction_length)
    with open(r"test_batch.pickle", "rb") as f:
        test_batch = pickle.load(f)


    if modeltype == "Transformer":
        path = rf"trained_model\Transformer_GMM_{encoder_length}_{n_component}_{n_hiddenlayer}_{hidden_size}_{epoch + 1}.pt"
        model = DeepK_GMM_Transformer(input_size, hidden_size, num_layers, prediction_length, spatial_dim, n_component, n_head, max_len).to(device)

    elif modeltype == "GCN":
        path = rf"trained_model\GCN_GMM_{encoder_length}_{n_component}_{n_hiddenlayer}_{hidden_size}_{epoch + 1}.pt"
        adj_matrix_np = generate_chain_adjacency(spatial_dim)
        adj = torch.from_numpy(adj_matrix_np).to(device)
        model = DeepK_GMM_GCN_LSTM(input_size,  hidden_size, num_layers,  prediction_length, spatial_dim, n_component, adj).to(device)

    else:
        pass

    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()

    with open(r"test_data_" + str(bus_route)+".pickle", "rb") as f:
        test_data = pickle.load(f)

    count = 0
    for date in test_data["date"].unique():
        data_day = test_data[test_data["date"] == date]
        data_day["time_idx"] = data_day["time_idx"] - data_day["time_idx"].min()
        test_batchx = test_batch[date + "x"]
        test_batchy = test_batch[date + "y"]
        pred_t = tstart
        time_span = 0.25
        t_max = data_day["tosan_observed_arrival_time_sfm"].max()/3600
        depart_t = data_day[data_day["group"] == "11.0"]["x"].values
        arrive_t = data_day[data_day["group"] == "40.0"]["tosan_observed_arrival_time_sfm"].values/3600

        while pred_t <= t_max:
            count+=1
            diff = arrive_t - pred_t
            diff[diff > 0] = np.inf
            completed_index = np.argmin(np.abs(diff))
            pot_uncompleted_trip_id = np.arange(completed_index + 1, completed_index + prediction_length + 1)
            if completed_index - encoder_length <= len(test_batchx)-1:

                x = test_batchx[completed_index - encoder_length].to(torch.float32)
                y = test_batchy[completed_index - encoder_length].to(torch.float32)
                y= y.permute(1, 0, 2)
                y = y.detach().numpy().reshape(-1)

                uncompleted_trip_id = []
                for pot_index in pot_uncompleted_trip_id:
                    if pot_index <= data_day["time_idx"].values.max():
                        if depart_t[pot_index] >= pred_t - 0.05:
                            pass
                        else:
                            uncompleted_trip_id.append(pot_index)

                for order, sel_trip in enumerate(uncompleted_trip_id):
                    stop_arrive_time = data_day[data_day["time_idx"] == sel_trip]["tosan_observed_arrival_time_sfm"].values/3600
                    time_diff_start = stop_arrive_time - pred_t
                    time_diff_start[time_diff_start > 0] = np.inf
                    obs_loc_indexlast = np.argmin(np.abs(time_diff_start))
                    observed_loc_index = np.arange(0, obs_loc_indexlast + 1)

                    time_diff_end = stop_arrive_time - pred_t - time_span
                    time_diff_end[time_diff_end > 0] = np.inf
                    pred_loc_indexend = np.argmin(np.abs(time_diff_end))
                    predicted_loc_index = np.arange(obs_loc_indexlast + 1, pred_loc_indexend+1)
                    travel_time = y[order * batch_size:(order+1) * batch_size]

                    if order == 0:
                        observed_loc = observed_loc_index
                        predicted_loc = predicted_loc_index
                        observed_tt = travel_time[observed_loc_index]
                        unobserved_tt = travel_time[predicted_loc_index]
                        travel_time_all = travel_time
                    else:
                        observed_loc = np.concatenate((observed_loc, observed_loc_index + order * batch_size))
                        predicted_loc = np.concatenate((predicted_loc, predicted_loc_index + order * batch_size))
                        observed_tt = np.concatenate((observed_tt, travel_time[observed_loc_index]))
                        unobserved_tt = np.concatenate((unobserved_tt, travel_time[predicted_loc_index]))
                        travel_time_all = np.concatenate((travel_time_all, travel_time))

                links = batch_size * len(uncompleted_trip_id)
                unobserved_loc = np.array([i for i, e in enumerate(range(links)) if e not in observed_loc])
                new_loc = np.concatenate((unobserved_loc, observed_loc), axis=0)
                sel_loc = [i for i, e in enumerate(unobserved_loc) if e in predicted_loc]

                pi, pred_means, pred_covs = model(x)
                pred_dict = {}

                for k in range(len(pi)):
                    pred_mean = pred_means[k].detach().numpy().reshape(-1)
                    pred_cov = pred_covs[k].detach().numpy()
                    pi_k = pi[k].detach().numpy()
                    pred_mean_new = np.array([pred_mean[i] for i in new_loc])
                    pred_cov_new = np.array([[pred_cov[i][j] for j in new_loc] for i in new_loc])

                    unobserved_pred_mean = pred_mean_new[:len(unobserved_loc)]
                    unobserved_cov = pred_cov_new[:len(unobserved_loc), :len(unobserved_loc)]
                    observed_cov = pred_cov_new[len(unobserved_loc):, len(unobserved_loc):]
                    unobserved_observed_cov = pred_cov_new[:len(unobserved_loc), len(unobserved_loc):]

                    inv = np.linalg.solve(observed_cov, np.eye(observed_cov.shape[0]))
                    pred_unobserved_mean = unobserved_pred_mean + unobserved_observed_cov @ inv @ (observed_tt - pred_mean_new[len(unobserved_loc):])
                    pred_unobserved_cov = unobserved_cov - unobserved_observed_cov @ inv @ unobserved_observed_cov.T

                    sel_mu = mu[unobserved_loc][sel_loc]
                    sel_sigma = sigma[unobserved_loc][sel_loc]

                    if k==0:
                        pred_unobserved_mean_sel = pi_k*(pred_unobserved_mean[sel_loc] * sel_sigma + sel_mu)
                        real_unobserved_mean_sel = travel_time_all[unobserved_loc][sel_loc] * sel_sigma + sel_mu
                        pred_unobserved_cov_sel = np.array([[pred_unobserved_cov[ix][iy] for iy in sel_loc] for ix in sel_loc])
                    else:
                        pred_unobserved_mean_sel += pi_k*(pred_unobserved_mean[sel_loc] * sel_sigma + sel_mu)
                        real_unobserved_mean_sel = travel_time_all[unobserved_loc][sel_loc] * sel_sigma + sel_mu
                        pred_unobserved_cov_sel = np.array([[pred_unobserved_cov[ix][iy] for iy in sel_loc] for ix in sel_loc])

                    pred_dict.update({f"weight_{k}": pi_k,
                                     f"pred_unobserved_mean_sel_{k}": pred_unobserved_mean_sel,
                                     f"real_unobserved_mean_sel": real_unobserved_mean_sel,
                                     f"pred_unobserved_cov_sel_{k}": pred_unobserved_cov_sel})

                end_time = time.time()
                testing_seconds = start_time - end_time
                print(f"Prediction time: {testing_seconds:.5f} seconds, {testing_seconds / count:.5f}")

                crps_point = []
                p_risk_point1 = []
                p_risk_point2 = []

                for link in range(len(sel_loc)):
                    observed = pred_dict["real_unobserved_mean_sel"][link]
                    if len(pi) == 1:
                        weight = pred_dict["weight_0"]
                        predicted_mu =pred_dict["pred_unobserved_mean_sel_0"][link]
                        predicted_cov = np.power(sel_sigma[link], 2) * pred_dict["pred_unobserved_cov_sel_0"][link, link]
                        predicted_sigma = np.sqrt(predicted_cov)
                        norm_tt = (observed - predicted_mu) / predicted_sigma
                        crps = sel_sigma[link] * (norm_tt * (2 * norm.cdf(x=norm_tt) - 1) + 2 * norm.pdf(x=norm_tt) - math.pow(
                                math.pi, -0.5))
                        pvalue1 = norm.ppf(0.5, loc=predicted_mu, scale=sel_sigma[link])
                        pvalue2 = norm.ppf(0.9, loc=predicted_mu, scale=sel_sigma[link])
                        if pvalue1 > observed:
                            coeff1 = 1 - 0.5
                        else:
                            coeff1 = -1 * 0.5
                        if pvalue2 > observed:
                            coeff2 = 1 - 0.9
                        else:
                            coeff2 = -1 * 0.9
                        p_risk1 = (pvalue1 - observed) * coeff1
                        p_risk2 = (pvalue2 - observed) * coeff2

                    else:
                        if len(pi) == 2:
                            weight = np.array([pred_dict["weight_0"], pred_dict["weight_1"]])
                            predicted_mu = np.array([pred_dict["pred_unobserved_mean_sel_0"][link], pred_dict["pred_unobserved_mean_sel_1"][link]])
                            cov1 = np.power(sel_sigma[link], 2) * pred_dict["pred_unobserved_cov_sel_0"][link, link]
                            cov2 = np.power(sel_sigma[link], 2) * pred_dict["pred_unobserved_cov_sel_1"][link, link]
                            predicted_cov = np.array([cov1, cov2])
                            predicted_sigma = np.sqrt(predicted_cov)
                        else:
                            weight = np.array([pred_dict["weight_0"], pred_dict["weight_1"], pred_dict["weight_2"]])
                            predicted_mu = np.array([pred_dict["pred_unobserved_mean_sel_0"][link],
                                                     pred_dict["pred_unobserved_mean_sel_1"][link],
                                                     pred_dict["pred_unobserved_mean_sel_2"][link]])
                            cov1 = np.power(sel_sigma[link], 2) * pred_dict["pred_unobserved_cov_sel_0"][link, link]
                            cov2 = np.power(sel_sigma[link], 2) * pred_dict["pred_unobserved_cov_sel_1"][link, link]
                            cov3 = np.power(sel_sigma[link], 2) * pred_dict["pred_unobserved_cov_sel_2"][link, link]
                            predicted_cov = np.array([cov1, cov2, cov3])
                            predicted_sigma = np.sqrt(predicted_cov)

                        A = np.zeros(len(pi))
                        for k in range(len(pi)):
                            z = (observed - predicted_mu[k]) / predicted_sigma[k]
                            A[k] = 2 * predicted_sigma[k] * norm.pdf(z) + (observed - predicted_mu[k]) * (
                                        2 * norm.cdf(z) - 1)
                        term1 = np.sum(weight * A)

                        B = np.zeros((len(pi), len(pi)))
                        for i in range(len(pi)):
                            for j in range(len(pi)):
                                sij = np.sqrt(predicted_sigma[i] ** 2 + predicted_sigma[j] ** 2)
                                a = (predicted_mu[i] - predicted_mu[j]) / sij
                                B[i, j] = 2 * sij * norm.pdf(a) + (predicted_mu[i] - predicted_mu[j]) * (
                                            2 * norm.cdf(a) - 1)
                        term2 = 0.5 * np.sum(weight[:, None] * weight[None, :] * B)
                        crps = term1 - term2

                        def gmm_quantile(p, mus, sigmas, weights, xtol=1e-10, maxiter=100):
                            mus = np.array(mus, dtype=float).reshape(-1)
                            sigmas = np.array(sigmas, dtype=float).reshape(-1)
                            w = np.asarray(weights, dtype=float).reshape(-1)
                            w = w / w.sum()

                            F = lambda y: np.sum(w * norm.cdf((y - mus) / sigmas)) - p

                            eps = 1e-12
                            zlo, zhi = norm.ppf(eps), norm.ppf(1 - eps)
                            low = np.min(mus + sigmas * zlo)
                            high = np.max(mus + sigmas * zhi)

                            flo, fhi = F(low), F(high)

                            if not (flo <= 0 and fhi >= 0):
                                center = np.dot(w, mus)
                                width = max(10.0 * sigmas.max(), 1.0)
                                low, high = center - width, center + width
                            return bisect(F, low, high, xtol=xtol, maxiter=maxiter)

                        pvalue1 = gmm_quantile(0.5, predicted_mu, predicted_sigma, weight)
                        pvalue2 = gmm_quantile(0.9, predicted_mu, predicted_sigma, weight)

                        if pvalue1 > real_unobserved_mean_sel[link]:
                            p_risk1 = (1-0.5) * (pvalue1-real_unobserved_mean_sel[link])
                        else:
                            p_risk1 = 0.5 * (real_unobserved_mean_sel[link] - pvalue1)
                        if pvalue2 > real_unobserved_mean_sel[link]:
                            p_risk2 = (1-0.9) * (pvalue2-real_unobserved_mean_sel[link])
                        else:
                            p_risk2 = 0.9 * (real_unobserved_mean_sel[link] - pvalue2)

                    crps_point.append(crps)
                    p_risk_point1.append(p_risk1)
                    p_risk_point2.append(p_risk2)

            if pred_t == tstart:
                pred_day = pred_unobserved_mean_sel
                real_day = real_unobserved_mean_sel
                crps_point_day = np.array(crps_point)
                p_risk_day1 = np.array(p_risk_point1)
                p_risk_day2 = np.array(p_risk_point2)

            else:
                pred_day = np.concatenate((pred_day, pred_unobserved_mean_sel), axis=0)
                real_day = np.concatenate((real_day, real_unobserved_mean_sel), axis=0)
                crps_point_day = np.concatenate((crps_point_day, np.array(crps_point)), axis=0)
                p_risk_day1 = np.concatenate((p_risk_day1, np.array(p_risk_point1)), axis=0)
                p_risk_day2 = np.concatenate((p_risk_day2, np.array(p_risk_point2)), axis=0)
            pred_t = pred_t + time_span

        if date == test_date:
            pred_all = pred_day
            real_all = real_day
            crps_point_all = crps_point_day
            p_risk_all1 = p_risk_day1
            p_risk_all2 = p_risk_day2

        else:
            pred_all = np.concatenate([pred_all, pred_day], axis=0)
            real_all = np.concatenate([real_all, real_day], axis=0)
            crps_point_all = np.concatenate([crps_point_all, crps_point_day], axis=0)
            p_risk_all1 = np.concatenate([p_risk_all1, p_risk_day1], axis=0)
            p_risk_all2 = np.concatenate([p_risk_all2, p_risk_day2], axis=0)

    end_time = time.time()
    testing_seconds = end_time
    print(f"Prediction time: {testing_seconds:.5f} seconds, {testing_seconds / (test_data_day.shape[0] - reg_len):.5f}")

    mape = np.sum(abs(pred_all - real_all) / real_all) / len(real_all) * 100
    rmse = root_mean_squared_error(pred_all, real_all)
    crps_point_mean = np.sum(np.array(crps_point_all)) / len(crps_point_all)
    p_risk_mean1 = np.sum(np.array(p_risk_all1)) / len(p_risk_all1)
    p_risk_mean2 = np.sum(np.array(p_risk_all2)) / len(p_risk_all2)

    error_pd = pd.DataFrame(np.array([[-1, -1, -1, -1, -1]]), columns=["mape", "rmse", "crps_point", "p_risk1", "p_risk2"])
    error_pd["mape"] = mape
    error_pd["rmse"] = rmse
    error_pd["crps_point"] = crps_point_mean
    error_pd["p_risk1"] = p_risk_mean1
    error_pd["p_risk2"] = p_risk_mean2

    print("error_pd", error_pd)
    with open(r"error_model" + str(modelID) + ".pickle", "wb") as f:
        pickle.dump(error_pd, f)
    return None


start_time = time.time()
test_date = "2022-03-23"
# modeltype = "Transformer"
modeltype = "GCN"
n_component =2
bus_route = 1
tstart = 7
encoder_length = 8
prediction_length = 7
input_size = 1
num_layers = 1
hidden_size = 32
n_head = 4
max_len = 9
batch_size = 30
spatial_dim = 30

if modeltype == "Transformer":
    epoch = 38
elif modeltype == "GCN":
    epoch = 24
else:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mu, sigma = loaddata()
forecasting(mu, sigma, modeltype, n_component)