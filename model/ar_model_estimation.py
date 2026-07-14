# =============================================================================
# Lead model conceptualization and development (theory, methodology, and
# implementation), software architecture & model coding (all components),
# data acquisition & processing, team supervision: Michael Miess
# Initial software architecture development, first coding, technical consulting:
# Joel Foramitti
# Lead of energy modeling: Ansir Ilyas
# Lead of water modeling: Dan Wang
# Model consulting & support: Asjad Naqvi
# Project conceptualization, funding & supervision: Yoshihide Wada
# =============================================================================

from statsmodels.tsa.ar_model import AutoReg
import ruptures as rpt
import numpy as np
import pandas as pd


def identify_structural_break(timeseries, method="bottomup"):
    # Detect change points using the specified method
    if method == "pelt":
        algo = rpt.Pelt(model="l2").fit(timeseries.values)
        result = algo.predict(pen=15)
    # Binary Segmentation: This method recursively partitions the data to detect change points. It is simpler and faster but may not be as accurate as Pelt for detecting multiple change points.
    elif method == "binary":
        algo = rpt.Binseg(model="l2").fit(timeseries.values)
        result = algo.predict(pen=10)
    # Bottom-Up Segmentation: This method starts with many small segments and merges them to detect change points. It is useful for detecting larger break points.
    elif method == "bottomup":
        algo = rpt.BottomUp(model="l2").fit(timeseries.values)
        result = algo.predict(pen=50)
    # Window-Based Segmentation: This method uses a sliding window to detect change points. It is useful for detecting changes in the local structure of the time series.
    elif method == "window":
        algo = rpt.Window(model="l2").fit(timeseries.values)
        result = algo.predict(pen=10)
    # Dynamic Programming: This method uses dynamic programming to find the optimal segmentation of the time series. It is more computationally intensive but can be more accurate.
    elif method == "dynamic":
        algo = rpt.Dynp(model="l2").fit(timeseries.values)
        result = algo.predict(n_bkps=10)
    else:
        raise ValueError("Unsupported method: {}".format(method))

    return result


def determine_optimal_lag_order(timeseries, max_lags, begin_time_series_estimation):
    timeseries = pd.Series(timeseries)
    timeseries.index = pd.date_range(
        start='1969', periods=len(timeseries), freq='YE')
    restricted_timeseries = timeseries[begin_time_series_estimation:]
    best_aic = float("inf")
    best_bic = float("inf")
    best_lag_aic = 0
    best_lag_bic = 0

    for lag in range(1, max_lags + 1):
        model = AutoReg(restricted_timeseries, lags=lag)
        model_fit = model.fit()
        aic = model_fit.aic
        bic = model_fit.bic

        if aic < best_aic:
            best_aic = aic
            best_lag_aic = lag

        if bic < best_bic:
            best_bic = bic
            best_lag_bic = lag

    return best_lag_aic, best_lag_bic


def determine_optimal_lag_order_growthrates(timeseries, max_lags, begin_time_series_estimation):
    # Ensure the data is numeric
    timeseries = pd.to_numeric(timeseries, errors='coerce')
    # Drop any NaN values that may have been introduced
    timeseries = timeseries.dropna()
    timeseries = pd.Series(timeseries)
    timeseries.index = pd.date_range(
        start='1969', periods=len(timeseries), freq='YE')
    timeseries_growthrates = timeseries.pct_change().dropna()
    restricted_timeseries_growthrates = timeseries_growthrates[begin_time_series_estimation:]
    best_aic = float("inf")
    best_bic = float("inf")
    best_lag_aic = 0
    best_lag_bic = 0

    for lag in range(1, max_lags + 1):
        model = AutoReg(restricted_timeseries_growthrates, lags=lag)
        model_fit = model.fit()
        aic = model_fit.aic
        bic = model_fit.bic

        if aic < best_aic:
            best_aic = aic
            best_lag_aic = lag

        if bic < best_bic:
            best_bic = bic
            best_lag_bic = lag

    return best_lag_aic, best_lag_bic


def determine_optimal_lag_order_log(timeseries, max_lags, begin_time_series_estimation):
    # Ensure the data is numeric
    timeseries = pd.to_numeric(timeseries, errors='coerce')
    # Drop any NaN values that may have been introduced
    timeseries = timeseries.dropna()
    timeseries = pd.Series(timeseries)
    timeseries.index = pd.date_range(
        start='1969', periods=len(timeseries), freq='Y')
    log_timeseries = np.log(timeseries)
    restricted_timeseries_growthrates = log_timeseries[begin_time_series_estimation:]
    best_aic = float("inf")
    best_bic = float("inf")
    best_lag_aic = 0
    best_lag_bic = 0

    for lag in range(1, max_lags + 1):
        model = AutoReg(restricted_timeseries_growthrates, lags=lag)
        model_fit = model.fit()
        aic = model_fit.aic
        bic = model_fit.bic

        if aic < best_aic:
            best_aic = aic
            best_lag_aic = lag

        if bic < best_bic:
            best_bic = bic
            best_lag_bic = lag

    return best_lag_aic, best_lag_bic


def estimate_ar_model(timeseries, optimal_lag, begin_time_series_estimation):
    # Ensure the data is numeric
    timeseries = pd.to_numeric(timeseries, errors='coerce')
    # Drop any NaN values that may have been introduced
    timeseries = timeseries.dropna()
    timeseries = pd.Series(timeseries)
    start_year = 1969 if 1969 in timeseries.index else 1970
    timeseries.index = pd.date_range(
        start=f'{start_year}', periods=len(timeseries), freq='Y')
    restricted_timeseries = timeseries[begin_time_series_estimation:]
    model = AutoReg(restricted_timeseries, lags=optimal_lag)
    model_fit = model.fit()
    return model_fit


def estimate_ar_model_log(timeseries, optimal_lag, begin_time_series_estimation):
    # Ensure the data is numeric
    timeseries = pd.to_numeric(timeseries, errors='coerce')
    # Drop any NaN values that may have been introduced
    timeseries = timeseries.dropna()
    timeseries = pd.Series(timeseries)
    start_year = 1969 if 1969 in timeseries.index else 1970
    timeseries.index = pd.date_range(
        start=f'{start_year}', periods=len(timeseries), freq='Y')
    log_timeseries = np.log(timeseries)
    restricted_log_timeseries = log_timeseries[begin_time_series_estimation:]
    model = AutoReg(restricted_log_timeseries, lags=optimal_lag)
    model_fit = model.fit()

    return model_fit

# Takes a timeseries as inputs in level, converts it to growth rates, and then estimates the AR model
# Specify the begin_time_series_estimation to restrict the timeseries, e.g. to the last 20 years, with -20


def estimate_ar_model_growthrates(timeseries, optimal_lag, begin_time_series_estimation):
    # Ensure the data is numeric
    timeseries = pd.to_numeric(timeseries, errors='coerce')
    # Drop any NaN values that may have been introduced
    timeseries = timeseries.dropna()
    timeseries = pd.Series(timeseries)
    start_year = 1969 if 1969 in timeseries.index else 1970
    timeseries.index = pd.date_range(
        start=f'{start_year}', periods=len(timeseries), freq='YE')
    timeseries_growthrates = timeseries.pct_change().dropna()
    restricted_timeseries_growthrates = timeseries_growthrates[begin_time_series_estimation:]
    model = AutoReg(restricted_timeseries_growthrates, lags=optimal_lag)
    model_fit = model.fit()
    return model_fit


def forecast(model_fit, steps):
    forecasted_values = model_fit.predict(
        start=len(model_fit.fittedvalues), end=len(model_fit.fittedvalues) + steps)
    return forecasted_values
