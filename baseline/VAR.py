import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from scipy.stats import norm
import math
from sklearn.metrics import root_mean_squared_error
import pickle
import scipy.stats
import matplotlib.pyplot as plt
import random
import time
random.seed(42)
np.random.seed(42)

max_encoder_length = 8
max_prediction_length = 6
bus_route = 2
test_date = "2022-04-21"
tstart = 7
reg_len = 5
def var():
    start_time = time.time()
    with open(r"forecasting_linkID.pickle", "rb") as f:
        dict_date_trip = pickle.load(f)

    cols = ["date", "avl_segment_travel_time", "x", "group", "time_idx", "time_idx_new", "avl_time_on_stop_at_fromsan"]
    data = pd.read_csv(r"Jan2Mar_TT_normal" +str(bus_route) + "_train.csv", usecols=cols)
    data = data.rename(columns={"avl_segment_travel_time": "ACTTT", "time_idx": "busid", "time_idx_new": "time_idx",
                 "avl_time_on_stop_at_fromsan": "dwell_time"})
    data["ACTTT"] = data["ACTTT"] + data["dwell_time"]
    data_train = data[data["date"] < "2022-04-21"].reset_index(drop=True)
    data_test = data[data["date"] >= "2022-04-21"].reset_index(drop=True)

    for i in data_train["group"].unique():
        series = data_train[data_train["group"] == i]["ACTTT"].reset_index(drop=True)
        if i == data_train["group"].values[0]:
            new_train_data = series
        else:
            new_train_data = pd.concat([new_train_data, series], axis=1)

    new_train_data.columns = np.arange(1, len(data["group"].unique())+1)
    model = VAR(new_train_data)
    model_fit = model.fit(reg_len)

    end_time = time.time()
    testing_seconds = start_time - end_time
    print(f"Prediction time: {testing_seconds:.5f} seconds")

    count = 0
    for date in data_test["date"].unique():
        print("date", date)
        data_day = data_test[data_test["date"] == date].reset_index(drop=True)
        for i in data_day["group"].unique():
            series = data_day[data_day["group"] == i]["ACTTT"].reset_index(drop=True)
            if i == data_day["group"].values[0]:
                test_data_day = series
            else:
                test_data_day = pd.concat([test_data_day, series], axis=1)
        test_data_day.columns = np.arange(1, len(data["group"].unique())+1)
        for i in range(0, test_data_day.shape[0]-reg_len, 1):
            count += 1
            pred_mean = model_fit.forecast(test_data_day[i:i+reg_len].values, 1)[0]
            pred_sigma = model_fit.forecast_cov(1)[0]
            real_mean = test_data_day[i+reg_len:i+reg_len+1].values[0]
            if i + reg_len in list(dict_date_trip[date].keys()):
                sel_index = dict_date_trip[date][i + reg_len]
                pre_mean_sel = pred_mean[sel_index]
                pre_cov_sel = pred_sigma[-len(sel_index):, -len(sel_index):]
                real_mean_sel = real_mean[sel_index]
                crps_point = []
                p_risk_point = []
                p_risk_point1 = []
                for link in range(len(pre_mean_sel)):
                    mu = pre_mean_sel[link]
                    sigma = math.sqrt(pre_cov_sel[link, link])
                    norm_tt = (real_mean_sel[link] - mu) / sigma
                    crps1 = sigma * (
                            norm_tt * (2 * norm.cdf(x=norm_tt) - 1) + 2 * norm.pdf(x=norm_tt) - math.pow(math.pi, -0.5))
                    crps_point.append(crps1)
                    pvalue = norm.ppf(0.5, loc=mu, scale=sigma)
                    pvalue1 = norm.ppf(0.9, loc=mu, scale=sigma)
                    if pvalue > real_mean_sel[link]:
                        coeff = 1 - 0.5
                    else:
                        coeff = -1 * 0.5
                    p_risk = 2 * (pvalue-real_mean_sel[link]) * coeff
                    p_risk_point.append(p_risk)

                    if pvalue1 > real_mean_sel[link]:
                        coeff1 = 1 - 0.9
                    else:
                        coeff1 = -1 * 0.9
                    p_risk1 = 2 * (pvalue1 - real_mean_sel[link]) * coeff1
                    p_risk_point1.append(p_risk1)
                if i + reg_len == list(dict_date_trip[date].keys())[0]:
                    pred_day = pre_mean_sel
                    real_day = real_mean_sel
                    crps_point_day = np.array(crps_point)
                    p_risk_day = np.array(p_risk_point)
                    p_risk_day1 = np.array(p_risk_point1)
                else:
                    pred_day = np.concatenate((pred_day, pre_mean_sel), axis=0)
                    real_day = np.concatenate((real_day, real_mean_sel), axis=0)
                    crps_point_day = np.concatenate((crps_point_day, np.array(crps_point)), axis=0)
                    p_risk_day = np.concatenate((p_risk_day, np.array(p_risk_point)), axis=0)
                    p_risk_day1 = np.concatenate((p_risk_day1, np.array(p_risk_point1)), axis=0)
            else:
                pass
        if date == "2022-04-21":
            pred_all = pred_day
            real_all = real_day
            crps_point_all = crps_point_day
            p_risk_all = p_risk_day
            p_risk_all1 = p_risk_day1
        else:
            pred_all = np.concatenate((pred_all, pred_day), axis=0)
            real_all = np.concatenate((real_all, real_day), axis=0)
            crps_point_all = np.concatenate((crps_point_all, crps_point_day), axis=0)
            p_risk_all = np.concatenate((p_risk_all, p_risk_day), axis=0)
            p_risk_all1 = np.concatenate((p_risk_all1, p_risk_day1), axis=0)

    end_time = time.time()
    testing_seconds = start_time - end_time
    print(f"Prediction time: {testing_seconds:.5f} seconds, {testing_seconds / count:.5f}")

    mape = np.sum(abs(pred_all - real_all) / real_all) / len(real_all) * 100
    rmse = root_mean_squared_error(pred_all, real_all)
    crps_point_mean = np.sum(np.array(crps_point_all)) / len(crps_point_all)
    p_risk_mean = np.sum(np.array(p_risk_all)) / len(p_risk_all)
    p_risk_mean1 = np.sum(np.array(p_risk_all1)) / len(p_risk_all1)
    print("mape", mape)
    print("rmse", rmse)
    print("crps point", len(crps_point_all), crps_point_mean)
    print("p-0.5risk", len(p_risk_all), p_risk_mean)
    print("p-0.9risk", len(p_risk_all1), p_risk_mean1)
    return None

var()