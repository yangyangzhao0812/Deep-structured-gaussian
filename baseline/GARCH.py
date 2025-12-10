import os
os.chdir("../../..")
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import root_mean_squared_error
plt.rcParams["font.family"] = "Times New Roman"
from matplotlib.pyplot import cm
from sklearn import preprocessing
import scipy
from scipy.stats import norm
import math
from scipy import linalg
from feature_engine.creation import CyclicalFeatures
from arch import arch_model
import mgarch
import random
import time
bus_route = 2

def garch():
    start_time = time.time()
    with open(r"forecasting_linkID.pickle", "rb") as f:
        dict_date_trip = pickle.load(f)

    cols = ["date", "avl_segment_travel_time", "x", "group", "time_idx", "time_idx_new", "avl_time_on_stop_at_fromsan"]
    data = pd.read_csv(r"Jan2Mar_TT_normal" +str(bus_route) + "_train.csv", usecols=cols)
    data = data.rename(
        columns={"avl_segment_travel_time": "ACTTT", "time_idx": "busid", "time_idx_new": "time_idx",
                 "avl_time_on_stop_at_fromsan": "dwell_time"})
    data["ACTTT"] = data["ACTTT"] + data["dwell_time"]

    test_data = data[data["date"] >= "2022-04-21"]
    data_row_dict = {"2022-04-21":0, "2022-04-22":0, "2022-04-25":0, "2022-04-26":0, "2022-04-27":0, "2022-04-28":0}
    for date in ["2022-04-21", "2022-04-22", "2022-04-25", "2022-04-26", "2022-04-27", "2022-04-28"]:
        date_sel = test_data[test_data["date"] == date]
        row_sel = date_sel[date_sel["group"] == 11].shape[0]
        data_row_dict[date] = row_sel


    date23 = data_row_dict["2022-04-21"] # number of trips
    date2324 = data_row_dict["2022-04-21"] + data_row_dict["2022-04-22"]
    date2328 = data_row_dict["2022-04-21"] + data_row_dict["2022-04-22"] + data_row_dict["2022-04-25"]
    date2329 = data_row_dict["2022-04-21"] + data_row_dict["2022-04-22"] + data_row_dict["2022-04-25"] +  data_row_dict["2022-04-26"]
    date2330 = data_row_dict["2022-04-21"] + data_row_dict["2022-04-22"] + data_row_dict["2022-04-25"] +  data_row_dict["2022-04-26"] + data_row_dict["2022-04-27"]
    date2331 = data_row_dict["2022-04-21"] + data_row_dict["2022-04-22"] + data_row_dict["2022-04-25"] + data_row_dict[
        "2022-04-26"] + data_row_dict["2022-04-27"] + data_row_dict["2022-04-28"]

    cutoff = data[data["date"] == "2022-04-21"]["time_idx"].values[0] - data["time_idx"].values[0]
    rows = data[data["group"] == 11].shape[0]
    tt_matrix = np.zeros((rows, 21))

    min_id = data["time_idx"].values[0]
    for i in range(rows):
        tt_matrix[i, :] = data[data["time_idx"] == min_id + i]["ACTTT"].values
    vol = mgarch.mgarch()
    date = "2022-04-21"

    vol.fit(tt_matrix[0:0 + cutoff, :])
    end_time1 = time.time()
    training_duration_seconds = end_time1 - start_time
    print(f"Total Model Training Time: {training_duration_seconds:.5f} seconds")

    pred_day = []
    real_day = []
    crps_point = []
    p_risk_point1 = []
    p_risk_point2 = []
    count = 0
    for id, i in enumerate(range(0, tt_matrix.shape[0] - cutoff)):
        if i <= 8:
            pass
        else:
            ndays = 1
            cov_nextday = vol.predict(ndays)
            pred_mean = np.array(cov_nextday["mean"]).reshape(-1)
            pred_sigma = cov_nextday["cov"]
            real_mean = np.array(tt_matrix[i + cutoff:i + cutoff + 1, :]).reshape(-1)
            if id < date23:
                date = "2022-04-21"
                id = id
            elif date23 <= id < date2324:
                date = "2022-04-22"
                id = id - date23
            elif date2324 <= id < date2328:
                date = "2022-04-25"
                id = id - date2324
            elif date2328 <= id < date2329:
                date = "2022-04-26"
                id = id - date2328
            elif date2329 <= id < date2330:
                date = "2022-04-27"
                id = id - date2329
            elif date2330 <= id < date2331:
                date = "2022-04-28"
                id = id - date2330
            else:
                pass
            if id in list(dict_date_trip[date].keys()):
                count += 1
                sel_index = dict_date_trip[date][id]
                pre_mean_sel = pred_mean[sel_index]
                pre_cov_sel = pred_sigma[-len(sel_index):, -len(sel_index):]
                real_mean_sel = real_mean[sel_index]
                for link in range(len(pre_mean_sel)):
                    mu = pre_mean_sel[link]
                    sigma = math.sqrt(pre_cov_sel[link, link])
                    norm_tt = (real_mean_sel[link] - mu) / sigma
                    crps1 = sigma * (
                            norm_tt * (2 * norm.cdf(x=norm_tt) - 1) + 2 * norm.pdf(x=norm_tt) - math.pow(math.pi, -0.5))
                    crps_point.append(crps1)

                    pvalue1 = norm.ppf(0.5, loc=mu, scale=sigma)
                    if pvalue1 > real_mean_sel[link]:
                        coeff1 = 1 - 0.5
                    else:
                        coeff1 = -1 * 0.5
                    p_risk1 = 2 * (pvalue1 - real_mean_sel[link]) * coeff1
                    p_risk_point1.append(p_risk1)

                    pvalue2 = norm.ppf(0.9, loc=mu, scale=sigma)
                    if pvalue2 > real_mean_sel[link]:
                        coeff2 = 1 - 0.9
                    else:
                        coeff2 = -1 * 0.9
                    p_risk2 = 2 * (pvalue2 - real_mean_sel[link]) * coeff2
                    p_risk_point2.append(p_risk2)
                if count == 0 :
                    pred_day = pre_mean_sel
                    real_day = real_mean_sel
                else:
                    pred_day = np.concatenate((pred_day, pre_mean_sel), axis=0)
                    real_day = np.concatenate((real_day, real_mean_sel), axis=0)
            else:
                pass

    end_time2 = time.time()
    testing_seconds = end_time2 - end_time1

    print(f"Prediction time: {testing_seconds:.5f} seconds, {testing_seconds / id:.5f}")

    mape = np.sum(abs(pred_day - real_day) / real_day) / len(real_day) * 100
    rmse = root_mean_squared_error(pred_day, real_day)
    crps_point_mean = np.sum(np.array(crps_point)) / len(crps_point)
    p_risk_mean1 = np.sum(np.array(p_risk_point1)) / len(p_risk_point1)
    p_risk_mean2 = np.sum(np.array(p_risk_point2)) / len(p_risk_point2)
    print("mape", len(real_day), mape)
    print("rmse", len(real_day), rmse)
    print("crps point", len(crps_point), crps_point_mean)
    print("p-0.5risk", len(p_risk_point1), p_risk_mean1)
    print("p-0.9risk", len(p_risk_point2), p_risk_mean2)
    return None

garch()