import numpy as np
import pandas as pd
from statsmodels.tsa.tsatools import vech, unvech
from scipy.linalg import sqrtm
from scipy.stats import t
from scipy.optimize import fmin_slsqp
import matplotlib.pyplot as plt


def compute_covariance(r):
    cov = r.groupby(r.index.date).apply(lambda x: x.T.dot(x))
    return cov


def decompose_covariance(cov_array):
    cov_vector = []
    for i in range(cov_array.shape[0]):
        cov_vector.append(vech(np.linalg.cholesky(sqrtm(cov_array[i]))))
    return np.array(cov_vector)


def generate_sequences(cov_vector, additional_features=True, features=None, window=60):
    X, X_feature, y = [], [], []

    for i in range(cov_vector.shape[0] - window):
        X.append(cov_vector[i:i + window])
        y.append(cov_vector[i + window])

        if additional_features:
            if type(features) is pd.DataFrame:
                X_feature.append(features.iloc[i:i + window])
            else:
                X_feature.append(features[i:i + window])

    if additional_features:
        return np.array(X), np.array(X_feature), np.array(y)

    return np.array(X), np.array(y)


def reconstruct_covariance(x):
    mat = unvech(x)  # undo vector operator
    mat = mat.dot(mat.T)  # undo Cholesky decomposition
    return mat.dot(mat)  # undo square root


def portfolio_opt(weights, cov, alpha=0.01, nu=3, opt_type='risk'):
    sig = np.sqrt(weights.dot(cov).dot(weights.T))
    if opt_type == 'risk':
        return sig
    x = t.ppf(alpha, nu)
    return - 1 / alpha * 1 / (1 - nu) * (nu - 2 + x**2) + t.pdf(x, nu) * sig


class Portfolio:
    def __init__(self, daily_returns, covariance, weight_constraint=1):
        self.weights = None
        self.portfolio_returns = None
        self.portfolio_log_returns = None
        self.sharpe = None
        self.sortino = None
        self.overall_return = None
        self.history = None
        self.daily_returns = daily_returns
        self.covariance = covariance
        self.constraint = weight_constraint
        self.N = daily_returns.shape[1]
        self.tickers = daily_returns.columns
        self.dates = daily_returns.index

    def optimize(self, objective='risk', alpha=0.01, nu=3, return_percent=True):
        bounds = [(0.0, 1.0)] * self.N
        equality_constraint = [lambda a, b, c, d, e: 1.0 - np.sum(a)]
        x0 = [1 / self.N] * self.N
        weights = []

        for i in range(self.covariance.shape[0]):
            result = fmin_slsqp(portfolio_opt,
                                x0=x0,
                                bounds=bounds,
                                eqcons=equality_constraint,
                                args=(self.covariance[i], alpha, nu, objective,),
                                iprint=False)
            weights.append(result)

        self.weights = pd.DataFrame(weights, columns=self.tickers, index=self.dates)
        self.portfolio_log_returns = (self.weights * self.daily_returns).sum(axis=1)
        self.portfolio_returns = np.exp(self.portfolio_log_returns) - 1

        if return_percent:
            return self.portfolio_returns

        return self.portfolio_log_returns

    def metrics(self, starting_value=100, print_metrics=True, plot_history=True):
        cumulative = (self.portfolio_returns + 1).cumprod()
        self.overall_return = cumulative.values[-1]
        self.history = starting_value * cumulative
        self.sharpe = sharpe_ratio(self.portfolio_returns)
        self.sortino = sortino_ratio(self.portfolio_returns)

        if print_metrics:
            print('Final Portfolio Value: ${}'.format(round(self.history.values[-1], 1)))
            print('Overall Return (%):', round((self.history.values[-1] - starting_value) / starting_value * 100, 2))
            print('Sharpe Ratio:', round(self.sharpe, 2))
            print('Sortino Ratio:', round(self.sortino, 2))

        if plot_history:
            plt.figure(figsize=(12, 8))
            plt.plot(self.history)
            plt.ylabel('Portfolio Value ($)')
            plt.title('Portfolio history starting with ${} investment'.format(str(starting_value)))


def sharpe_ratio(returns):
    mean_return = returns.mean()
    risk = returns.std()
    return mean_return / risk * np.sqrt(252)


def sortino_ratio(returns):
    mean_return = returns.mean()
    downside_risk = returns[returns < 0].std()
    return mean_return / downside_risk * np.sqrt(252)
