# %%
import pandas as pd
import numpy as np
import pyswarms as ps
np.set_printoptions(precision=4, suppress=True)
N_ACTIONS = 6
N_LAGS = 3
NEURONS = 10


class Optimizador:
    def __init__(self) -> None:
        pass

    def set_dataset(self, dataset):
        if(len(dataset) == 0):
            return False
        self.dataset = dataset
        return True

    def clean_dataset(self, path):
        stock_price = pd.read_excel(path)
        stock_price['Data'] = stock_price['Data'].str.replace(r'Dez', 'Dec')
        stock_price['Data'] = stock_price['Data'].str.replace(r'Fev', 'Feb')
        stock_price['Data'] = stock_price['Data'].str.replace(r'Abr', 'Apr')
        stock_price['Data'] = stock_price['Data'].str.replace(r'Mai', 'May')
        stock_price['Data'] = stock_price['Data'].str.replace(r'Ago', 'Aug')
        stock_price['Data'] = stock_price['Data'].str.replace(r'Set', 'Sep')
        stock_price['Data'] = stock_price['Data'].str.replace(r'Out', 'Oct')
        stock_price.set_index(pd.to_datetime(
            stock_price["Data"].values, format="%b-%Y"))
        stock_price = stock_price.drop(columns="Data")
        return stock_price

    def split_dataset(self, dataset, n_splits=2):
        datasets = []
        portion = int(len(dataset)/n_splits)

        tmp_dataset = dataset.iloc[:portion, :]
        datasets.append(tmp_dataset)
        index = portion
        for i in range(1, n_splits):
            if(i == n_splits-1):
                tmp_dataset = dataset.iloc[index:, :]
            else:
                tmp_dataset = dataset.iloc[index:index+portion, :]
                index += portion
            datasets.append(tmp_dataset)

        self.datasets = datasets
        return datasets

    def lag_variables(self, dataset, n_lag=2):
        lagged_dataset = []
        for index in range(n_lag+1, len(dataset)+1):
            tmp = dataset.iloc[index-n_lag-1: index,
                               :].pct_change(1)[1:].values.flatten()
            lagged_dataset.append(tmp)
        columns = [dataset.columns]*(n_lag)
        for index in range(n_lag-1):
            columns[index] = columns[index]+f"-{n_lag-index-1}"
        lagged_dataset = pd.DataFrame(
            lagged_dataset, columns=np.array(columns).flatten())
        return lagged_dataset

    def nn_structure(self, lagged_dataset, w):
        result = lagged_dataset @ np.reshape(
            w[:N_ACTIONS*N_LAGS*NEURONS], (N_ACTIONS*N_LAGS, NEURONS))
        result = np.tanh(result)
        result = result @ np.reshape(w[N_ACTIONS *
                                     N_LAGS*NEURONS:], (NEURONS, N_ACTIONS))
        result[result < 0] = 0
        for index in range(len(result)):
            result.iloc[index] = result.iloc[index] / \
                (np.sum(result.iloc[index])+1e-15)
        return result

    def calc_ret_vol(self, ativos, port_pesos):
        sharpe, _, _ = self.calc_cash(ativos, port_pesos)
        return sharpe

    def calc_cash(self, ativos, port_pesos):
        cash = 100.0
        ativos_chg = ativos.pct_change()
        tmp_variations = []
        cash_memory = [cash, ]
        for index in range(N_LAGS, len(ativos)):
            tmp_chg = ativos_chg.iloc[index].values
            tmp_pesos = port_pesos.iloc[index-N_LAGS].values
            cash_variation = tmp_chg @  tmp_pesos
            cash = cash*(1+cash_variation)
            cash_memory.append(cash)
            tmp_variations.append(cash_variation)
        variation_estable = np.array(
            tmp_variations) - ativos_chg.iloc[N_LAGS:, -3]
        rets = variation_estable.mean()*252
        vols = variation_estable.std()*np.sqrt(252)
        sharpe = rets/(vols*2)
        return sharpe, cash, cash_memory

    def _optimization(self, w, lagged_dataset, dataset):
        result = self.nn_structure(lagged_dataset, w)
        return -1*self.calc_ret_vol(dataset, result)

    def optimization(self, W, lagged_dataset, dataset):
        result = []
        for w in W:
            result.append(self._optimization(w, lagged_dataset, dataset))
        return result

    def plot(self, dataset, variations):
        data = dataset.copy()
        data = data.iloc[N_LAGS-1:]
        data = data / data.iloc[0]*100
        data["Thal-ia"] = variations
        print(data.tail())
        data[N_LAGS-1:].plot()

    def run(self, l_dataset, dataset):
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = ps.single.GlobalBestPSO(
            n_particles=20, dimensions=NEURONS*N_ACTIONS*(N_LAGS+1), options=options)
        return optimizer.optimize(self.optimization, iters=100, lagged_dataset=l_dataset, dataset=dataset, n_processes=4)


if __name__ == "__main__":
    opt = Optimizador()
    dataset = opt.clean_dataset("../../datasets/Dados-Trabalho-2004-2015.xlsx")
    datasets = opt.split_dataset(dataset,3)
    dataset_lagged = opt.lag_variables(datasets[0], N_LAGS)
    cost, pos = opt.run(dataset_lagged, datasets[0])

    # %%
    # np.save("best_weights2", pos)
    pesos = opt.nn_structure(dataset_lagged, pos)
    sharpe, cash, variations = opt.calc_cash(datasets[0], pesos)
    opt.plot(datasets[0], variations)
    sharpe, cash
    # %%
    dataset_lagged_2 = opt.lag_variables(datasets[1], N_LAGS)
    pesos = opt.nn_structure(dataset_lagged_2, pos)
    sharpe, cash, variations = opt.calc_cash(datasets[1], pesos)
    opt.plot(datasets[1], variations)
    sharpe, cash
    # %%
    dataset_main_lagged = opt.lag_variables(dataset, N_LAGS)
    pesos = opt.nn_structure(dataset_main_lagged, pos)
    sharpe, cash, variations = opt.calc_cash(dataset, pesos)
    opt.plot(dataset, variations)
    sharpe, cash