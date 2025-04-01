
import os
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool
from itertools import chain, combinations
from functools import partial
from math import factorial as fac
from scipy.stats import norm
import copy
from tqdm import tqdm
from sklearn.linear_model import Ridge, LinearRegression
import bisect
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ensure multiprocessing uses 'fork' method
multiprocessing.set_start_method('fork', force=True)

# Helper functions
number_to_letter = {
    '01': 'A',
    '06': 'B',
    '07': 'C',
    '11': 'D'
}

def check_xlsx(file_list):
    """Check for .xlsx files in the list and warn if found."""
    xlsx_files = [file for file in file_list if file.endswith('.xlsx')]
    if xlsx_files:
        print("The following .xlsx files were found:")
        for file_ in xlsx_files:
            print(file_)
        print("--> Please save all .xlsx files as .csv files to ensure faster data import!")
        return True
    return False

def load_scada_data(data_path: str,
                    lst_files = ['Wind-Turbine-SCADA-signals-2016.csv', 
                                 'Wind-Turbine-SCADA-signals-2017_0.csv']) -> dict:
    """Load and process SCADA data from multiple CSV files.

    Args:
        base_path (str): Directory path containing CSV files (with trailing /).
        lst_files (list): List of filenames to load.
        
    Returns:
        dict: Mapping of turbine IDs to their DataFrames with UTC timestamps.
    """
    print('Loading SCADA signals...')
    dct_scada = {}

    if check_xlsx(lst_files):
        return {}

    df_scada_all = pd.DataFrame()
    for file in lst_files:
        df_temp = pd.read_csv(os.path.join(data_path, file), index_col='Timestamp')
        df_temp.index = pd.to_datetime(df_temp.index, utc=True)
        df_scada_all = pd.concat([df_scada_all, df_temp])

    for trb_id in np.unique(df_scada_all.Turbine_ID):
        df_turbine = df_scada_all[df_scada_all['Turbine_ID'] == trb_id]
        df_turbine = df_turbine.drop('Turbine_ID', axis='columns')
        df_turbine.index.name = trb_id
        df_turbine['power'] = df_turbine.Grd_Prod_Pwr_Avg
        df_turbine['wind_speed'] = df_turbine.Amb_WindSpeed_Avg
        df_turbine['wind_direction'] = df_turbine.Amb_WindDir_Abs_Avg

        turbine_number = trb_id.split('_')[0][1:]
        turbine_letter = number_to_letter.get(turbine_number, turbine_number)
        new_key = f'Turbine_{turbine_letter}'

        dct_scada[new_key] = df_turbine[~df_turbine.index.duplicated(keep='first')].sort_index()

    return dct_scada

def load_metmast_data(data_path=str,
                      lst_files=['2016_WF1edp_metmast.csv', '2017_WF1edp_metmast.csv']):
    """Load and process met mast data from CSV files.

    Args:
        base_path (str): Directory path containing CSV files.
        lst_files (list): List of filenames to load.
        
    Returns:
        pd.DataFrame: Processed met mast data.
    """
    df_metmast_all = pd.concat([pd.read_csv(os.path.join(data_path, file), delimiter=';') for file in lst_files])
    df_metmast_all.index = pd.to_datetime(df_metmast_all['Timestamp'], utc=True)
    df_metmast_all = df_metmast_all.drop('Timestamp', axis='columns').sort_index()

    df_metmast_all = df_metmast_all[df_metmast_all.Var_Windspeed2 < 25]

    R = 287.05  # J/(kg*K)
    p = df_metmast_all['Avg_Pressure'] * 100  # from mbar to N/m²
    T = df_metmast_all['Avg_AmbientTemp'] + 273.15
    roh_0 = 1.225  # kg/m³

    df_metmast_all['roh_air'] = p / (R * T)
    h_rel = df_metmast_all['Avg_Humidity'] / 100
    p_w = 0.0000205 * np.exp(0.0631846 * T)
    R_w = 461.5
    p_corr_h = 0
    roh_iec = (1 / T) * (((p - p_corr_h) / R) - (h_rel * p_w * ((1 / R) - (1 / R_w))))
    df_metmast_all['rho_iec'] = roh_iec

    return df_metmast_all

def fill_dct_data(dct_data_empty, dct_scada, df_metmast):
    """Populate the empty data dictionary with relevant SCADA and met mast data.

    Args:
        dct_data_empty (dict): Dictionary containing empty datasets to be filled.
        dct_scada (dict): Dictionary containing SCADA data for each turbine.
        df_metmast (DataFrame): DataFrame containing met mast data.

    Returns:
        dict: Populated data dictionary with SCADA and met mast data.
    """
    for trb_id, datasets in dct_data_empty.items():
        for set_name, data in datasets.items():
            if set_name in ['scaler', 'df_pc']:
                continue

            idx = data.index
            scada_data = dct_scada[trb_id].loc[idx]
            data['windspeed_trb'] = scada_data['Amb_WindSpeed_Avg']
            data['ti_trb'] = scada_data['Amb_WindSpeed_Std'] / scada_data['Amb_WindSpeed_Avg']
            data['output_trb'] = scada_data['Grd_Prod_Pwr_Avg']
            data['rho_iec'] = df_metmast.loc[idx, 'rho_iec']

    dct_data_empty['lst_inputs'] = ['windspeed_trb', 'rho_iec', 'ti_trb']
    dct_data_empty['target'] = 'output_trb'
    print('... data loaded, selected, and organized!')
    return dct_data_empty

def get_Xy(dct_data, trb_id_, set_, normalized=False):
    """Extract features and target variable from the data dictionary.

    Args:
        dct_data (dict): Data dictionary.
        trb_id_ (str): Turbine ID.
        set_ (str): Dataset name.
        normalized (bool): Whether to normalize the input features.

    Returns:
        tuple: Features (X) and target variable (y).
    """
    y = dct_data[trb_id_][set_][dct_data['target']]
    if normalized:
        X = dct_data[trb_id_][set_][dct_data['lst_inputs']]
        X = dct_data[trb_id_]['scaler'].transform(X)
    else:
        X = np.array(dct_data[trb_id_][set_][dct_data['lst_inputs']])
    return X, y

class PHYSbase:
    """Physics-based model for wind turbine power prediction."""

    def __init__(self, df_pc, rho_ref, ti_adjust=[0.5, 0.066, 0.819]):
        self.df_pc = df_pc
        self.rho_ref = rho_ref
        self.ti_cap, self.ti_shift, self.ti_scale = ti_adjust

    def predict_single(self, data_point, rho_corr=True, ti_corr=True):
        """Predict power output for a single data point.

        Args:
            data_point (array-like): Input data point [wind_speed, rho, ti].
            rho_corr (bool): Whether to apply air density correction.
            ti_corr (bool): Whether to apply turbulence intensity correction.

        Returns:
            np.array: Predicted power output.
        """
        v_w, roh, ti = np.round(data_point[0], 2), data_point[1], data_point[2]

        if not rho_corr and not ti_corr:
            return np.array(self.df_pc.loc[v_w, 'power']).clip(0, 2000000)

        if rho_corr:
            v_w = np.round(v_w * (roh / self.rho_ref) ** (1 / 3), 2)
            p_rho = self.df_pc.loc[v_w, 'power']

        if not ti_corr:
            return np.array(p_rho).clip(0, 2000000)
        
        # ti-correction
        
        # ----------------------------------------------------------------------- #
        # TI-pre-processing:
        # "For the nacelle-measured TI values to better match the TI
        # distribution of the nearby met mast, we apply a simple pruning and bias
        # correction." (SEC 4.1 - page 5):
        #
        # From plotting the distributions of tubine TI values against the (un-
        # disturbed ones from the metmast, we ovserved that a simple scale, shift 
        # and clip of turbine-based TI-values (obtained by iterative optimization) 
        # results in a much better alignment of the two. Data-driven methods can
        # learn this transformation from the data directly - therefore it is not
        # applied to any of the other methods.

        ti = min(self.ti_cap, max(0.0001, (ti * self.ti_scale - self.ti_shift)))
        pdf_act = norm(v_w, max(0.00001, ti * v_w))
        v_min, v_max = np.round(max(0, pdf_act.ppf(0.01)), 2), np.round(pdf_act.ppf(0.99), 2)

        if v_min == v_max:
            p_ti = self.df_pc.loc[v_w, 'power']
        else:
            lst_v = [np.round(i, 2) for i in np.arange(v_min, v_max, 0.05)]
            p_ti = np.sum([pdf_act.pdf(v_) * self.df_pc.loc[v_.clip(0, 99.9), 'power'] for v_ in lst_v])
            p_ti /= np.sum([pdf_act.pdf(v_) for v_ in lst_v])

        return np.array(p_ti).clip(0, 2000000)

    def predict(self, X, rho_corr=True, ti_corr=True):
        """Predict power output for multiple data points.

        Args:
            X (array-like): Input data points.
            rho_corr (bool): Whether to apply air density correction.
            ti_corr (bool): Whether to apply turbulence intensity correction.

        Returns:
            np.array: Predicted power outputs.
        """
        X = np.array(X)
        partial_mod = partial(self.predict_single, rho_corr=rho_corr, ti_corr=ti_corr)
        with Pool(processes=8) as pool:
            lst_output_pwr = np.array(list(tqdm(pool.imap(partial_mod, X), total=len(X))))
        return lst_output_pwr

### !!! old pyhs_base-class (depreciated) !!! ###
class PHYS_base:
    """Physics-based model for wind turbine power prediction with IEC corrections."""

    def __init__(self, df_pc_trb_rho, df_pc_null_adjusted, rho_ref):
        self.df_pc_trb_rho = df_pc_trb_rho
        self.df_pc_null_adjusted = df_pc_null_adjusted
        self.rho_ref = rho_ref

    def predict(self, data_point, rho_corr=True, ti_corr=True):
        """Predict power output for a single data point.

        Args:
            data_point (array-like): Input data point [wind_speed, rho, ti].
            rho_corr (bool): Whether to apply air density correction.
            ti_corr (bool): Whether to apply turbulence intensity correction.

        Returns:
            np.array: Predicted power output.
        """
        v_w, rho, ti = np.round(data_point[0], 2), data_point[1], data_point[2]

        if rho_corr:
            v_w = np.round(v_w * (rho / self.rho_ref) ** (1 / 3), 2)

        pred_pc_trb_rho = self.df_pc_trb_rho.loc[v_w, 'power']

        if not ti_corr:
            return np.array(pred_pc_trb_rho).clip(0, 2000000)

        pdf_ref = norm(v_w, max(0.001, self.df_pc_trb_rho.loc[np.round(v_w, 1)].TI) * v_w)
        v_start, v_stop = np.round(max(0, pdf_ref.ppf(0.001)), 1), np.round(pdf_ref.ppf(0.999), 1)

        if v_start == v_stop:
            p_ti_ref = self.df_pc_null_adjusted.loc[v_w, 'power']
        else:
            lst_v = [np.round(i, 2) for i in np.arange(v_start, v_stop, 0.05)]
            p_ti_ref = np.sum([pdf_ref.pdf(v_) * self.df_pc_null_adjusted.loc[v_.clip(0, 99.9), 'power'] for v_ in lst_v])
            p_ti_ref /= np.sum([pdf_ref.pdf(v_) for v_ in lst_v])

        pdf_act = norm(v_w, max(0.001, ti * v_w))
        v_start, v_stop = np.round(max(0, pdf_act.ppf(0.001)), 1), np.round(pdf_act.ppf(0.999), 1)

        if v_start == v_stop:
            p_ti_act = self.df_pc_null_adjusted.loc[v_w, 'power']
        else:
            lst_v = [np.round(i, 2) for i in np.arange(v_start, v_stop, 0.05)]
            p_ti_act = np.sum([pdf_act.pdf(v_) * self.df_pc_null_adjusted.loc[v_.clip(0, 99.9), 'power'] for v_ in lst_v])
            p_ti_act /= np.sum([pdf_act.pdf(v_) for v_ in lst_v])

        p_res = pred_pc_trb_rho - p_ti_ref + p_ti_act
        return np.array(p_res).clip(0, 2000000)

class PieceWiseModel:
    """Piecewise linear or polynomial regression model."""

    def __init__(self, lst_segments, poly=False):
        """Initialize the PieceWiseModel.

        Args:
            lst_segments (list): List of segment boundaries.
            poly (bool): Whether to use polynomial features.
        """
        self.lst_segments = lst_segments
        self.dct_models = {}
        self.poly = poly

    def fit(self, X_, y_, scaler):
        """Fit the piecewise model.

        Args:
            X_ (array-like): Input features.
            y_ (array-like): Target values.
            scaler (MinMaxScaler): Scaler for normalizing segment boundaries.
        """
        dummy_input = np.column_stack([self.lst_segments, np.zeros(len(self.lst_segments)), np.zeros(len(self.lst_segments))])
        self.lst_segments_normed = scaler.transform(dummy_input)[:, 0]

        if self.poly:
            X_ = np.concatenate([X_, X_**2, X_**3], axis=1)

        for i in tqdm(range(1, len(self.lst_segments_normed))):
            msk_low = X_[:, 0] > self.lst_segments_normed[i - 1]
            msk_high = X_[:, 0] < self.lst_segments_normed[i]
            msk_ = msk_low & msk_high

            X_train = X_[msk_]
            y_train = y_[msk_]

            model_ = Ridge(0.01) if self.poly else LinearRegression()

            if len(X_train) == 0:
                self.dct_models[i - 1] = None
                continue

            model_.fit(X_train, y_train)
            self.dct_models[i - 1] = model_

    def predict(self, X):
        """Predict using the fitted piecewise model.

        Args:
            X (array-like): Input data for prediction.

        Returns:
            np.array: Predicted values.
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        lst_pred = []

        for i, X_ in enumerate(X):
            if self.poly:
                X_ = np.concatenate([X_.reshape(-1, 3), X_.reshape(-1, 3)**2, X_.reshape(-1, 3)**3], axis=1)
                v_seg = X_[0, 0]
            else:
                v_seg = X_[0]

            index_seg = bisect.bisect_left(self.lst_segments_normed, v_seg) - 1

            if (index_seg not in self.dct_models.keys()) or (self.dct_models[index_seg] is None):
                print(f'No appropriate model for input {i}:')
                if X_[i, 0] > 12:
                    lst_pred.append(2000)
                    print(f'v_w above v_rated - therefore p_max predicted')
                elif X_[i, 0] < 4:
                    lst_pred.append(0)
                    print(f'v_w below v_rated - therefore 0 predicted')
                else:
                    lst_pred.append(np.nan)
                    print(f'NaN predicted')
            else:
                lst_pred.append(self.dct_models[index_seg].predict(X_.reshape(1, -1)))

        return np.array(lst_pred).flatten()

class shapley_sklearn_wrapper:
    """Wrapper for sklearn models to use with Shapley values."""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X.reshape(1, -1))

def powerset(iterable):
    """Generate the power set of an iterable."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def ff(x, msk, x_ref=np.array([0])):
    """Flip function for Shapley values."""
    xx = copy.copy(x)
    xx[msk] = x_ref[msk] if not np.all(x_ref == np.array([0])) else 0
    return xx

def shapley(x, f, flip_fun=ff, x_ref=np.array([0])):
    """Compute Shapley values for a given input and model."""
    n = len(x)
    R = np.zeros(x.shape)
    results = {}
    for i in range(n):
        N_i = [j for j in range(n) if j != i]
        for coalition in powerset(N_i):
            amount = fac(n - 1 - len(coalition)) * fac(len(coalition))
            msk = np.array([True] * n)
            msk[list(coalition)] = False

            if coalition in results:
                f1 = results[coalition]
            else:
                x1 = flip_fun(x, msk, x_ref)
                f1 = f(x1)
                results[coalition] = f1

            msk[i] = False
            coalition = tuple(sorted(coalition + (i,)))

            if coalition in results:
                f2 = results[coalition]
            else:
                x2 = flip_fun(x, msk, x_ref)
                f2 = f(x2)
                results[coalition] = f2

            R[i] += amount * (f2 - f1)

    return R / fac(n)

def explain_model(model, X, X_ref=[]):
    """Explain the model using Shapley values.

    Args:
        model: The model to explain.
        X (array-like): Input data.
        X_ref (array-like): Reference data for Shapley values.

    Returns:
        pd.DataFrame: DataFrame containing Shapley values.
    """
    if 'PHYSbase' not in str(type(model)):
        model_ = shapley_sklearn_wrapper(model)
        f_pred = model_.predict
    else:
        f_pred = model.predict_single

    if len(X_ref) == 0:
        X_ref = X.min(0)

    partial_shapley = partial(shapley, f=f_pred, x_ref=np.array(X_ref))
    with Pool(processes=8) as pool:
        lst_exp = np.array(list(tqdm(pool.imap(partial_shapley, X), total=len(X))))

    return pd.DataFrame(lst_exp, columns=model.input_features)

def plot_model_strategy(df_exp, v_w, lst_ax=[], c='darkgrey', alpha=0.2, hatch=''):
    """Plot the model strategy.

    Args:
        df_exp (pd.DataFrame): DataFrame containing explanations.
        v_w (array-like): Wind speeds.
        lst_ax (list): List of axes for plotting.
        c (str): Color for the plot.
        alpha (float): Transparency level.
        hatch (str): Hatch pattern for the plot.

    Returns:
        list: List of axes with the plot.
    """
    font = {'size': 20}
    plt.rc('font', **font)

    n_params = df_exp.shape[1]
    v_w = np.round(v_w * 2) / 2

    if not lst_ax:
        fig, ax = plt.subplots(ncols=n_params, figsize=(22, 7))
    else:
        ax = lst_ax[0]

    for j in range(n_params):
        mean_ = df_exp.groupby(v_w).median().iloc[:, j]
        min_ = df_exp.groupby(v_w).min().iloc[:, j]
        max_ = df_exp.groupby(v_w).max().iloc[:, j]
        x_ = mean_.index

        if c == 'darkgrey':
            ax[j].plot(mean_, label=df_exp.columns[j], c='k')
            ax[j].plot(x_, min_, color='k', alpha=alpha * 0.75)
            ax[j].plot(x_, max_, color='k', alpha=alpha * 0.75)
            ax[j].fill_between(x_, min_, max_, color=c, alpha=alpha, hatch=hatch)
        else:
            ax[j].plot(mean_, label=df_exp.columns[j], c=c)
            ax[j].plot(x_, min_, color=c, alpha=1)
            ax[j].plot(x_, max_, color=c, alpha=1)
            ax[j].fill_between(x_, min_, max_, color=c, alpha=alpha, hatch=hatch)

        ax[j].axhline(0, c='grey')
        ax[j].grid(True)
        ax[j].set_title(df_exp.columns[j])

    ax[0].set_ylabel('$P(R_i \mid v_{w,i})$ [kW]')
    ax[1].set_xlabel('Wind speed ($v_w$) [m/s]')

    return ax

class PlotUpdater:
    """Class to update plots based on dropdown selections."""

    def __init__(self, dct_models, dct_data):
        self.dct_models = dct_models
        self.dct_data = dct_data
        self.setup_dropdowns()

    def setup_dropdowns(self):
        """Set up dropdowns for model and turbine selection."""
        self.dropdown_trb_1 = widgets.Dropdown(options=['Turbine_A', 'Turbine_B', 'Turbine_C', 'Turbine_D'], value='Turbine_A', description='Turbine:')
        self.dropdown_model_1 = widgets.Dropdown(options=['ANNlarge', 'ANNsmall', 'PHYSbase', 'PLR', 'PPR', 'RF', 'SVR'], value='ANNlarge', description='Model:')
        self.dropdown_i_model_1 = widgets.Dropdown(options=list(range(len(self.dct_models[f'Turbine_A_ANNlarge']))), value=0, description='i_model:')

        self.dropdown_trb_2 = widgets.Dropdown(options=['Turbine_A', 'Turbine_B', 'Turbine_C', 'Turbine_D'], value='Turbine_A', description='Turbine:')
        self.dropdown_model_2 = widgets.Dropdown(options=['ANNlarge', 'ANNsmall', 'PHYSbase', 'PLR', 'PPR', 'RF', 'SVR'], value='ANNlarge', description='Model:')
        self.dropdown_i_model_2 = widgets.Dropdown(options=list(range(len(self.dct_models[f'Turbine_A_ANNlarge']))), value=0, description='i_model:')

        self.dropdown_trb_1.observe(self.update_i_model_dropdown, names='value')
        self.dropdown_model_1.observe(self.update_i_model_dropdown, names='value')
        self.dropdown_i_model_1.observe(self.update_plot, names='value')

        self.dropdown_trb_2.observe(self.update_i_model_dropdown, names='value')
        self.dropdown_model_2.observe(self.update_i_model_dropdown, names='value')
        self.dropdown_i_model_2.observe(self.update_plot, names='value')

        column1 = widgets.VBox([widgets.HTML("<h3 style='font-weight: bold; color: darkgreen; font-size: 40px;'>Model A</h3>"), self.dropdown_trb_1, self.dropdown_model_1, self.dropdown_i_model_1])
        column2 = widgets.VBox([widgets.HTML("<h3 style='font-weight: bold; color: blue; font-size: 40px;'>Model B</h3>"), self.dropdown_trb_2, self.dropdown_model_2, self.dropdown_i_model_2])
        self.layout = widgets.HBox([column1, column2])
        self.output = widgets.Output()

    def update_i_model_dropdown(self, change):
        """Update the i_model dropdown based on the selected model."""
        trb_1 = self.dropdown_trb_1.value
        model_1 = self.dropdown_model_1.value
        trb_2 = self.dropdown_trb_2.value
        model_2 = self.dropdown_model_2.value

        self.dropdown_i_model_1.options = list(range(len(self.dct_models[f'{trb_1}_{model_1}'])))
        self.dropdown_i_model_2.options = list(range(len(self.dct_models[f'{trb_2}_{model_2}'])))

        self.dropdown_i_model_1.value = 0
        self.dropdown_i_model_2.value = 0

        self.update_plot(None)

    def update_plot(self, change):
        """Update the plot based on the selected models and turbines."""
        with self.output:
            self.output.clear_output(wait=True)

            trb_1 = self.dropdown_trb_1.value
            model_1 = self.dropdown_model_1.value
            i_model_1 = self.dropdown_i_model_1.value

            trb_2 = self.dropdown_trb_2.value
            model_2 = self.dropdown_model_2.value
            i_model_2 = self.dropdown_i_model_2.value

            model_A = self.dct_models[f'{trb_1}_{model_1}'][i_model_1]
            model_B = self.dct_models[f'{trb_2}_{model_2}'][i_model_2]

            v_w_1 = get_Xy(self.dct_data, trb_1, 'val', normalized=False)[0][:, 0]
            v_w_2 = get_Xy(self.dct_data, trb_2, 'val', normalized=False)[0][:, 0]

            fig, ax = plt.subplots(figsize=(12, 4), ncols=3)
            ax = plot_model_strategy(model_A.df_exp, v_w_1, lst_ax=[ax], c='darkgreen')
            ax = plot_model_strategy(model_B.df_exp, v_w_2, lst_ax=[ax], c='blue')

            fig, ax = plt.subplots(figsize=(12, 6), ncols=2)

            ax[0].set_title("Strategy vs. Error")
            ax[0].axhline(1, ls='--', c='grey')
            ax[0].axvline(1, ls='--', c='grey')

            ax[0].set_xlabel('$R^2_{Strategy PHYSbase}$')
            ax[0].set_ylabel('$RMSE_{test}$ [rel. to PHYSbase]')

            rmse_phys_a = self.dct_models[f'{trb_1}_PHYSbase'][0].df_rmse['test']
            rmse_phys_b = self.dct_models[f'{trb_1}_PHYSbase'][0].df_rmse['test']

            ax[0].scatter([model_A.df_corr_exp['weighted_mean']], [model_A.df_rmse['test'] / rmse_phys_a],
                          marker='X', s=500, c='darkgreen', alpha=0.8)
            ax[0].scatter([model_B.df_corr_exp['weighted_mean']], [model_B.df_rmse['test'] / rmse_phys_b],
                          marker='X', s=500, c='blue', alpha=0.8)

            ax[0].grid(True)
            
            legend_ax = ax[1]
            legend_ax.axis('off')

            patch_a = mpatches.Patch(color='darkgreen', label=f'{trb_1}_{model_1}')
            patch_b = mpatches.Patch(color='blue', label=f'{trb_2}_{model_2}')

            rmse_a, r_a = np.round(model_A.df_rmse['test'], 2), np.round(model_A.df_corr_exp['weighted_mean'], 2)
            text_a, text_b = f'$RMSE_A$ = {rmse_a}kW', f'$R^2_A$ = {r_a}'

            rmse_b, r_b = np.round(model_B.df_rmse['test'], 2), np.round(model_B.df_corr_exp['weighted_mean'], 2)
            text_c, text_d = f'$RMSE_B$ = {rmse_b}kW', f'$R^2_B$ = {r_b}'

            legend_ax.legend(handles=[mpatches.Patch(color='none', label=text_a),
                                      mpatches.Patch(color='none', label=text_b),
                                      mpatches.Patch(color='none', label=''),
                                      mpatches.Patch(color='none', label=text_c),
                                      mpatches.Patch(color='none', label=text_d)],
                             loc='center')

            plt.tight_layout()
            plt.show()
