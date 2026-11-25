import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return float(np.mean(0.01 * (u / d)))


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def SMAPE(pred, true, eps=1e-6):
    den = np.abs(true) + np.abs(pred) + eps
    return np.mean(2.0 * np.abs(pred - true) / den)


def QL(pred, true, q=0.5):
    err = true - pred
    return np.mean(np.maximum(q * err, (q - 1.0) * err))


def PW_MAE(pred, true, q=0.9):
    threshold = np.quantile(np.abs(true), q)
    weights = np.where(np.abs(true) >= threshold, 2.0, 1.0)
    return np.mean(weights * np.abs(pred - true))


def MASE(pred, true, inputx=None, eps=1e-6):
    if inputx is None:
        scale = np.mean(np.abs(np.diff(true, axis=1)))
    else:
        hist = inputx
        if hist.shape[0] != true.shape[0]:
            n = min(hist.shape[0], true.shape[0])
            hist = hist[:n]
            pred = pred[:n]
            true = true[:n]
        scale = np.mean(np.abs(hist[:, 1:, :] - hist[:, :-1, :]))
    scale = max(float(scale), eps)
    return np.mean(np.abs(pred - true)) / scale


def metric(pred, true, inputx=None):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    smape = SMAPE(pred, true)
    mase = MASE(pred, true, inputx=inputx)
    ql50 = QL(pred, true, q=0.5)
    pwmae = PW_MAE(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr, smape, mase, ql50, pwmae
