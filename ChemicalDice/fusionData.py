import os
import re
from sklearn.impute import KNNImputer, SimpleImputer
# Import the necessary modules

from sklearn.preprocessing import normalize

from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.model_selection import KFold
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

from sklearn.decomposition import PCA, FastICA
from sklearn.cross_decomposition import PLSRegression, CCA, PLSCanonical
from sklearn.decomposition import IncrementalPCA

from itertools import combinations
from sklearn.cross_decomposition import CCA

from sklearn.decomposition import KernelPCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.manifold import Isomap, TSNE, SpectralEmbedding, LocallyLinearEmbedding

from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import itertools

from functools import reduce
import pandas as pd
from ChemicalDice.plot_data import *
from ChemicalDice.preprocess_data import *
from ChemicalDice.saving_data import *
from ChemicalDice.analyse_data import *

import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib.colors import to_rgba

# Import the necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#########################

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List
import os
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from ChemicalDice.getEmbeddings import AutoencoderReconstructor_training_other , AutoencoderReconstructor_testing, AutoencoderReconstructor_training_8192, AutoencoderReconstructor_training_single
import time
from ChemicalDice.splitting import random_scaffold_split_train_val_test, scaffold_split_balanced_train_val_test, scaffold_split_train_val_test


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern


class fusionData:
    """
    Class for handling data and performing fusion tasks.
    Initialize fusionData object with provided data paths, label file path, ID column name,
    and prediction label column name.

    Args:
        data_paths (dict): Dictionary of file paths containing the data.
        label_file_path (str): file paths containing the label data.
        id_column (str, optional): Name of the column representing unique identifiers.
            Defaults to "id".
        prediction_label_column (str, optional): Name of the column containing prediction labels.
            Defaults to "labels".

    Attributes:
        - data_paths (list): List of file paths containing the data.
        - id_column (str): Name of the column representing unique identifiers.
        - dataframes (dict): Dictionary containing processed dataframes.
        - prediction_label: Placeholder for labels.


    """
    def __init__(self, data_paths, label_file_path, id_column = "id", label_column = "labels"):

        self.data_paths = data_paths
        loaded_data_dict = clear_and_process_data(data_paths, id_column)



        label_df = pd.read_csv(label_file_path)
        label_df.set_index(id_column, inplace=True)
        self.prediction_label = label_df[label_column]
        self.smiles_col = label_df["SMILES"]

        #check_missing_values(loaded_data_dict)
        #self.prediction_label = None
        self.dataframes = loaded_data_dict


        self.dataframes_transformed = False
        self.olddataframes = None
        
        self.top_features = None
        self.fusedData = None
        self.scaling_done = False
        
        self.pls_model = None
        self.Accuracy_metrics=None
        self.mean_accuracy_metrics =None
        self.train_dataframes = None
        self.test_dataframes = None
        self.train_label = None
        self.test_label = None
        self.training_AER_model = None
        self.AER_model_embt_size = None
        self.AER_model_explainability = None

    
    
    def ShowMissingValues(self):
        """
        Prints missing values inside dataframes in fusionData object.

        """
        dataframes = self.dataframes
        # Iterate through the dictionary and print missing DataFrame 
        for name, df in dataframes.items():
            missing_values = df.isnull().sum().sum()
            print(f"Dataframe name: {name}\nMissing values: {missing_values}\n")

    def show_info(self):
        """
        Summarize dataframes inside the fusionData object.

        """
        show_dataframe_info(self.dataframes)
    
    def plot_info(self, about, save_dir=None):
        """

        Generate plots for visualizing missing values and to check for normality of data.

        This method supports two types of plots:
           - 'missing_values': will generate a plot showing the missing values in the dataframes.
           - 'normality': will generate a bar plot to check the normality of the data in the dataframes.

        :param about: The topic for the plot. Must be either 'missing_values' or 'normality'.
        :type about: str

        """
        if about == "missing_values":
            plot_missing_values(list(self.dataframes.values()),
                                list(self.dataframes.keys()),save_dir)
        elif about == "normality":
            barplot_normality_check(list(self.dataframes.values()),
                                list(self.dataframes.keys()),save_dir)
        else:
            raise ValueError("Please select a valid plot type: 'missing_values' or 'normality'")

    def keep_common_samples(self):
        """
        Keep only the common samples or rows using the id in dataframes.

        """
        # Create an empty set to store the common indices
        common_indices = set()
        dataframes = self.dataframes
        # Loop through the df_dict
        for name, df in dataframes.items():
            # Get the index of the current dataframe and convert it to a set
            index = set(df.index)

            # Update the common indices set with the intersection of the current index
            if not common_indices:
                common_indices = index
            else:
                common_indices = common_indices.intersection(index)

        # Create an empty dictionary to store the filtered dataframes
        filtered_df_dict = {}

        # Loop through the df_dict again
        for name, df in dataframes.items():
            # Filter the dataframe by the common indices and store it in the filtered_df_dict
            filtered_df = df.loc[list(common_indices)]
            filtered_df_dict[name] = filtered_df

        self.prediction_label = self.prediction_label.loc[list(common_indices)]
        self.smiles_col = self.smiles_col.loc[list(common_indices)]
        self.dataframes = filtered_df_dict


        
    def remove_empty_features(self,threshold=100):
        
        """
        Remove columns with more than a threshold percentage of missing values from dataframes.

        :param threshold: The percentage threshold of missing values to drop a column. It should be between 0 and 100.
        :type threshold: float

        """
        dataframes = self.dataframes
        for name, df in dataframes.items():
            if name == 'mordred':
                common_columns = ['GeomDiameter', 'GeomRadius', 'GeomShapeIndex', 'GeomPetitjeanIndex', 'IC0', 'IC1', 'IC2', 'IC3', 'IC4', 'IC5', 'TIC0', 'TIC1', 'TIC2', 'TIC3', 'TIC4', 'TIC5', 'SIC0', 'SIC1', 'SIC2', 'SIC3', 'SIC4', 'SIC5', 'BIC0', 'BIC1', 'BIC2', 'BIC3', 'BIC4', 'BIC5', 'CIC0', 'CIC1', 'CIC2', 'CIC3', 'CIC4', 'CIC5', 'MIC0', 'MIC1', 'MIC2', 'MIC3', 'MIC4', 'MIC5', 'ZMIC0', 'ZMIC1', 'ZMIC2', 'ZMIC3', 'ZMIC4', 'ZMIC5', 'nRing', 'n3Ring', 'n4Ring', 'n5Ring', 'n6Ring', 'n7Ring', 'n8Ring', 'n9Ring', 'n10Ring', 'n11Ring', 'n12Ring', 'nG12Ring', 'nHRing', 'n3HRing', 'n4HRing', 'n5HRing', 'n6HRing', 'n7HRing', 'n8HRing', 'n9HRing', 'n10HRing', 'n11HRing', 'n12HRing', 'nG12HRing', 'naRing', 'n3aRing', 'n4aRing', 'n5aRing', 'n6aRing', 'n7aRing', 'n8aRing', 'n9aRing', 'n10aRing', 'n11aRing', 'n12aRing', 'nG12aRing', 'naHRing', 'n3aHRing', 'n4aHRing', 'n5aHRing', 'n6aHRing', 'n7aHRing', 'n8aHRing', 'n9aHRing', 'n10aHRing', 'n11aHRing', 'n12aHRing', 'nG12aHRing', 'nARing', 'n3ARing', 'n4ARing', 'n5ARing', 'n6ARing', 'n7ARing', 'n8ARing', 'n9ARing', 'n10ARing', 'n11ARing', 'n12ARing', 'nG12ARing', 'nAHRing', 'n3AHRing', 'n4AHRing', 'n5AHRing', 'n6AHRing', 'n7AHRing', 'n8AHRing', 'n9AHRing', 'n10AHRing', 'n11AHRing', 'n12AHRing', 'nG12AHRing', 'nFRing', 'n4FRing', 'n5FRing', 'n6FRing', 'n7FRing', 'n8FRing', 'n9FRing', 'n10FRing', 'n11FRing', 'n12FRing', 'nG12FRing', 'nFHRing', 'n4FHRing', 'n5FHRing', 'n6FHRing', 'n7FHRing', 'n8FHRing', 'n9FHRing', 'n10FHRing', 'n11FHRing', 'n12FHRing', 'nG12FHRing', 'nFaRing', 'n4FaRing', 'n5FaRing', 'n6FaRing', 'n7FaRing', 'n8FaRing', 'n9FaRing', 'n10FaRing', 'n11FaRing', 'n12FaRing', 'nG12FaRing', 'nFaHRing', 'n4FaHRing', 'n5FaHRing', 'n6FaHRing', 'n7FaHRing', 'n8FaHRing', 'n9FaHRing', 'n10FaHRing', 'n11FaHRing', 'n12FaHRing', 'nG12FaHRing', 'nFARing', 'n4FARing', 'n5FARing', 'n6FARing', 'n7FARing', 'n8FARing', 'n9FARing', 'n10FARing', 'n11FARing', 'n12FARing', 'nG12FARing', 'nFAHRing', 'n4FAHRing', 'n5FAHRing', 'n6FAHRing', 'n7FAHRing', 'n8FAHRing', 'n9FAHRing', 'n10FAHRing', 'n11FAHRing', 'n12FAHRing', 'nG12FAHRing', 'MOMI-X', 'MOMI-Y', 'MOMI-Z', 'VMcGowan', 'WPath', 'WPol', 'C1SP1', 'C2SP1', 'C1SP2', 'C2SP2', 'C3SP2', 'C1SP3', 'C2SP3', 'C3SP3', 'C4SP3', 'HybRatio', 'Zagreb1', 'Zagreb2', 'mZagreb1', 'mZagreb2', 'SpAbs_DzZ', 'SpMax_DzZ', 'SpDiam_DzZ', 'SpAD_DzZ', 'SpMAD_DzZ', 'LogEE_DzZ', 'SM1_DzZ', 'VE1_DzZ', 'VE2_DzZ', 'VE3_DzZ', 'VR1_DzZ', 'VR2_DzZ', 'VR3_DzZ', 'SpAbs_Dzm', 'SpMax_Dzm', 'SpDiam_Dzm', 'SpAD_Dzm', 'SpMAD_Dzm', 'LogEE_Dzm', 'SM1_Dzm', 'VE1_Dzm', 'VE2_Dzm', 'VE3_Dzm', 'VR1_Dzm', 'VR2_Dzm', 'VR3_Dzm', 'SpAbs_Dzv', 'SpMax_Dzv', 'SpDiam_Dzv', 'SpAD_Dzv', 'SpMAD_Dzv', 'LogEE_Dzv', 'SM1_Dzv', 'VE1_Dzv', 'VE2_Dzv', 'VE3_Dzv', 'VR1_Dzv', 'VR2_Dzv', 'VR3_Dzv', 'SpAbs_Dzse', 'SpMax_Dzse', 'SpDiam_Dzse', 'SpAD_Dzse', 'SpMAD_Dzse', 'LogEE_Dzse', 'SM1_Dzse', 'VE1_Dzse', 'VE2_Dzse', 'VE3_Dzse', 'VR1_Dzse', 'VR2_Dzse', 'VR3_Dzse', 'SpAbs_Dzpe', 'SpMax_Dzpe', 'SpDiam_Dzpe', 'SpAD_Dzpe', 'SpMAD_Dzpe', 'LogEE_Dzpe', 'SM1_Dzpe', 'VE1_Dzpe', 'VE2_Dzpe', 'VE3_Dzpe', 'VR1_Dzpe', 'VR2_Dzpe', 'VR3_Dzpe', 'SpAbs_Dzare', 'SpMax_Dzare', 'SpDiam_Dzare', 'SpAD_Dzare', 'SpMAD_Dzare', 'LogEE_Dzare', 'SM1_Dzare', 'VE1_Dzare', 'VE2_Dzare', 'VE3_Dzare', 'VR1_Dzare', 'VR2_Dzare', 'VR3_Dzare', 'SpAbs_Dzp', 'SpMax_Dzp', 'SpDiam_Dzp', 'SpAD_Dzp', 'SpMAD_Dzp', 'LogEE_Dzp', 'SM1_Dzp', 'VE1_Dzp', 'VE2_Dzp', 'VE3_Dzp', 'VR1_Dzp', 'VR2_Dzp', 'VR3_Dzp', 'SpAbs_Dzi', 'SpMax_Dzi', 'SpDiam_Dzi', 'SpAD_Dzi', 'SpMAD_Dzi', 'LogEE_Dzi', 'SM1_Dzi', 'VE1_Dzi', 'VE2_Dzi', 'VE3_Dzi', 'VR1_Dzi', 'VR2_Dzi', 'VR3_Dzi', 'nAtom', 'nHeavyAtom', 'nSpiro', 'nBridgehead', 'nH', 'nB', 'nC', 'nN', 'nO', 'nS', 'nP', 'nF', 'nCl', 'nBr', 'nI', 'nX', 'MDEC-22', 'MDEC-23', 'MDEC-33', 'BalabanJ', 'ETA_alpha', 'AETA_alpha', 'ETA_shape_p', 'ETA_shape_y', 'ETA_shape_x', 'ETA_beta', 'AETA_beta', 'ETA_beta_s', 'AETA_beta_s', 'ETA_beta_ns', 'AETA_beta_ns', 'ETA_beta_ns_d', 'AETA_beta_ns_d', 'ETA_eta', 'AETA_eta', 'ETA_eta_L', 'AETA_eta_L', 'ETA_eta_R', 'AETA_eta_R', 'ETA_eta_RL', 'AETA_eta_RL', 'ETA_eta_F', 'AETA_eta_F', 'ETA_eta_FL', 'AETA_eta_FL', 'ETA_eta_B', 'AETA_eta_B', 'ETA_eta_BR', 'AETA_eta_BR', 'ETA_dAlpha_A', 'ETA_dAlpha_B', 'ETA_epsilon_1', 'ETA_epsilon_2', 'ETA_epsilon_3', 'ETA_epsilon_4', 'ETA_epsilon_5', 'ETA_dEpsilon_A', 'ETA_dEpsilon_B', 'ETA_dEpsilon_C', 'ETA_dEpsilon_D', 'ETA_dBeta', 'AETA_dBeta', 'ETA_psi_1', 'ETA_dPsi_A', 'ETA_dPsi_B', 'MID', 'AMID', 'MID_h', 'AMID_h', 'MID_C', 'AMID_C', 'MID_N', 'AMID_N', 'MID_O', 'AMID_O', 'MID_X', 'AMID_X', 'GGI1', 'GGI2', 'GGI3', 'GGI4', 'GGI5', 'GGI6', 'GGI7', 'GGI8', 'GGI9', 'GGI10', 'JGI1', 'JGI2', 'JGI3', 'JGI4', 'JGI5', 'JGI6', 'JGI7', 'JGI8', 'JGI9', 'JGI10', 'JGT10', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10', 'SlogP_VSA11', 'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'EState_VSA10', 'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'fMF', 'SZ', 'Sm', 'Sv', 'Sse', 'Spe', 'Sare', 'Sp', 'Si', 'MZ', 'Mm', 'Mv', 'Mse', 'Mpe', 'Mare', 'Mp', 'Mi', 'nRot', 'RotRatio', 'NsLi', 'NssBe', 'NssssBe', 'NssBH', 'NsssB', 'NssssB', 'NsCH3', 'NdCH2', 'NssCH2', 'NtCH', 'NdsCH', 'NaaCH', 'NsssCH', 'NddC', 'NtsC', 'NdssC', 'NaasC', 'NaaaC', 'NssssC', 'NsNH3', 'NsNH2', 'NssNH2', 'NdNH', 'NssNH', 'NaaNH', 'NtN', 'NsssNH', 'NdsN', 'NaaN', 'NsssN', 'NddsN', 'NaasN', 'NssssN', 'NsOH', 'NdO', 'NssO', 'NaaO', 'NsF', 'NsSiH3', 'NssSiH2', 'NsssSiH', 'NssssSi', 'NsPH2', 'NssPH', 'NsssP', 'NdsssP', 'NsssssP', 'NsSH', 'NdS', 'NssS', 'NaaS', 'NdssS', 'NddssS', 'NsCl', 'NsGeH3', 'NssGeH2', 'NsssGeH', 'NssssGe', 'NsAsH2', 'NssAsH', 'NsssAs', 'NsssdAs', 'NsssssAs', 'NsSeH', 'NdSe', 'NssSe', 'NaaSe', 'NdssSe', 'NddssSe', 'NsBr', 'NsSnH3', 'NssSnH2', 'NsssSnH', 'NssssSn', 'NsI', 'NsPbH3', 'NssPbH2', 'NsssPbH', 'NssssPb', 'SsLi', 'SssBe', 'SssssBe', 'SssBH', 'SsssB', 'SssssB', 'SsCH3', 'SdCH2', 'SssCH2', 'StCH', 'SdsCH', 'SaaCH', 'SsssCH', 'SddC', 'StsC', 'SdssC', 'SaasC', 'SaaaC', 'SssssC', 'SsNH3', 'SsNH2', 'SssNH2', 'SdNH', 'SssNH', 'SaaNH', 'StN', 'SsssNH', 'SdsN', 'SaaN', 'SsssN', 'SddsN', 'SaasN', 'SssssN', 'SsOH', 'SdO', 'SssO', 'SaaO', 'SsF', 'SsSiH3', 'SssSiH2', 'SsssSiH', 'SssssSi', 'SsPH2', 'SssPH', 'SsssP', 'SdsssP', 'SsssssP', 'SsSH', 'SdS', 'SssS', 'SaaS', 'SdssS', 'SddssS', 'SsCl', 'SsGeH3', 'SssGeH2', 'SsssGeH', 'SssssGe', 'SsAsH2', 'SssAsH', 'SsssAs', 'SsssdAs', 'SsssssAs', 'SsSeH', 'SdSe', 'SssSe', 'SaaSe', 'SdssSe', 'SddssSe', 'SsBr', 'SsSnH3', 'SssSnH2', 'SsssSnH', 'SssssSn', 'SsI', 'SsPbH3', 'SssPbH2', 'SsssPbH', 'SssssPb', 'PNSA1', 'PNSA2', 'PNSA3', 'PNSA4', 'PNSA5', 'PPSA1', 'PPSA2', 'PPSA3', 'PPSA4', 'PPSA5', 'DPSA1', 'DPSA2', 'DPSA3', 'DPSA4', 'DPSA5', 'FNSA1', 'FNSA2', 'FNSA3', 'FNSA4', 'FNSA5', 'FPSA1', 'FPSA2', 'FPSA3', 'FPSA4', 'FPSA5', 'WNSA1', 'WNSA2', 'WNSA3', 'WNSA4', 'WNSA5', 'WPSA1', 'WPSA2', 'WPSA3', 'WPSA4', 'WPSA5', 'RNCG', 'RPCG', 'RNCS', 'RPCS', 'TASA', 'TPSA', 'RASA', 'RPSA', 'BertzCT', 'nBonds', 'nBondsO', 'nBondsS', 'nBondsD', 'nBondsT', 'nBondsA', 'nBondsM', 'nBondsKS', 'nBondsKD', 'GRAV', 'GRAVp', 'Xch-3d', 'Xch-4d', 'Xch-5d', 'Xch-6d', 'Xch-7d', 'Xch-3dv', 'Xch-4dv', 'Xch-5dv', 'Xch-6dv', 'Xch-7dv', 'Xc-3d', 'Xc-4d', 'Xc-5d', 'Xc-6d', 'Xc-3dv', 'Xc-4dv', 'Xc-5dv', 'Xc-6dv', 'Xpc-4d', 'Xpc-5d', 'Xpc-6d', 'Xpc-4dv', 'Xpc-5dv', 'Xpc-6dv', 'Xp-0d', 'Xp-1d', 'Xp-2d', 'Xp-3d', 'Xp-4d', 'Xp-5d', 'Xp-6d', 'Xp-7d', 'AXp-0d', 'AXp-1d', 'AXp-2d', 'AXp-3d', 'AXp-4d', 'AXp-5d', 'AXp-6d', 'AXp-7d', 'Xp-0dv', 'Xp-1dv', 'Xp-2dv', 'Xp-3dv', 'Xp-4dv', 'Xp-5dv', 'Xp-6dv', 'Xp-7dv', 'AXp-0dv', 'AXp-1dv', 'AXp-2dv', 'AXp-3dv', 'AXp-4dv', 'AXp-5dv', 'AXp-6dv', 'AXp-7dv', 'nAcid', 'nBase', 'ECIndex', 'SLogP', 'SMR', 'Diameter', 'Radius', 'TopoShapeIndex', 'PetitjeanIndex', 'SpAbs_D', 'SpMax_D', 'SpDiam_D', 'SpAD_D', 'SpMAD_D', 'LogEE_D', 'SM1_D', 'VE1_D', 'VE2_D', 'VE3_D', 'VR1_D', 'VR2_D', 'VR3_D', 'TopoPSA(NO)', 'TopoPSA', 'nAromAtom', 'nAromBond', 'MWC01', 'MWC02', 'MWC03', 'MWC04', 'MWC05', 'MWC06', 'MWC07', 'MWC08', 'MWC09', 'MWC10', 'TMWC10', 'SRW02', 'SRW03', 'SRW04', 'SRW05', 'SRW06', 'SRW07', 'SRW08', 'SRW09', 'SRW10', 'TSRW10', 'VAdjMat', 'MW', 'AMW', 'BCUTc-1h', 'BCUTc-1l', 'BCUTdv-1h', 'BCUTdv-1l', 'BCUTd-1h', 'BCUTd-1l', 'BCUTs-1h', 'BCUTs-1l', 'BCUTZ-1h', 'BCUTZ-1l', 'BCUTm-1h', 'BCUTm-1l', 'BCUTv-1h', 'BCUTv-1l', 'BCUTse-1h', 'BCUTse-1l', 'BCUTpe-1h', 'BCUTpe-1l', 'BCUTare-1h', 'BCUTare-1l', 'BCUTp-1h', 'BCUTp-1l', 'BCUTi-1h', 'BCUTi-1l', 'fragCpx', 'apol', 'bpol', 'SpAbs_A', 'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A', 'LogEE_A', 'SM1_A', 'VE1_A', 'VE2_A', 'VE3_A', 'VR1_A', 'VR2_A', 'VR3_A', 'Kier1', 'Kier2', 'Kier3', 'MPC2', 'MPC3', 'MPC4', 'MPC5', 'MPC6', 'MPC7', 'MPC8', 'MPC9', 'MPC10', 'TMPC10', 'piPC1', 'piPC2', 'piPC3', 'piPC4', 'piPC5', 'piPC6', 'piPC7', 'piPC8', 'piPC9', 'piPC10', 'TpiPC10', 'nHBAcc', 'nHBDon', 'ATS0dv', 'ATS1dv', 'ATS2dv', 'ATS3dv', 'ATS4dv', 'ATS5dv', 'ATS6dv', 'ATS7dv', 'ATS8dv', 'ATS0d', 'ATS1d', 'ATS2d', 'ATS3d', 'ATS4d', 'ATS5d', 'ATS6d', 'ATS7d', 'ATS8d', 'ATS0s', 'ATS1s', 'ATS2s', 'ATS3s', 'ATS4s', 'ATS5s', 'ATS6s', 'ATS7s', 'ATS8s', 'ATS0Z', 'ATS1Z', 'ATS2Z', 'ATS3Z', 'ATS4Z', 'ATS5Z', 'ATS6Z', 'ATS7Z', 'ATS8Z', 'ATS0m', 'ATS1m', 'ATS2m', 'ATS3m', 'ATS4m', 'ATS5m', 'ATS6m', 'ATS7m', 'ATS8m', 'ATS0v', 'ATS1v', 'ATS2v', 'ATS3v', 'ATS4v', 'ATS5v', 'ATS6v', 'ATS7v', 'ATS8v', 'ATS0se', 'ATS1se', 'ATS2se', 'ATS3se', 'ATS4se', 'ATS5se', 'ATS6se', 'ATS7se', 'ATS8se', 'ATS0pe', 'ATS1pe', 'ATS2pe', 'ATS3pe', 'ATS4pe', 'ATS5pe', 'ATS6pe', 'ATS7pe', 'ATS8pe', 'ATS0are', 'ATS1are', 'ATS2are', 'ATS3are', 'ATS4are', 'ATS5are', 'ATS6are', 'ATS7are', 'ATS8are', 'ATS0p', 'ATS1p', 'ATS2p', 'ATS3p', 'ATS4p', 'ATS5p', 'ATS6p', 'ATS7p', 'ATS8p', 'ATS0i', 'ATS1i', 'ATS2i', 'ATS3i', 'ATS4i', 'ATS5i', 'ATS6i', 'ATS7i', 'ATS8i', 'AATS0dv', 'AATS1dv', 'AATS2dv', 'AATS3dv', 'AATS4dv', 'AATS5dv', 'AATS6dv', 'AATS7dv', 'AATS0d', 'AATS1d', 'AATS2d', 'AATS3d', 'AATS4d', 'AATS5d', 'AATS6d', 'AATS7d', 'AATS0s', 'AATS1s', 'AATS2s', 'AATS3s', 'AATS4s', 'AATS5s', 'AATS6s', 'AATS7s', 'AATS0Z', 'AATS1Z', 'AATS2Z', 'AATS3Z', 'AATS4Z', 'AATS5Z', 'AATS6Z', 'AATS7Z', 'AATS0m', 'AATS1m', 'AATS2m', 'AATS3m', 'AATS4m', 'AATS5m', 'AATS6m', 'AATS7m', 'AATS0v', 'AATS1v', 'AATS2v', 'AATS3v', 'AATS4v', 'AATS5v', 'AATS6v', 'AATS7v', 'AATS0se', 'AATS1se', 'AATS2se', 'AATS3se', 'AATS4se', 'AATS5se', 'AATS6se', 'AATS7se', 'AATS0pe', 'AATS1pe', 'AATS2pe', 'AATS3pe', 'AATS4pe', 'AATS5pe', 'AATS6pe', 'AATS7pe', 'AATS0are', 'AATS1are', 'AATS2are', 'AATS3are', 'AATS4are', 'AATS5are', 'AATS6are', 'AATS7are', 'AATS0p', 'AATS1p', 'AATS2p', 'AATS3p', 'AATS4p', 'AATS5p', 'AATS6p', 'AATS7p', 'AATS0i', 'AATS1i', 'AATS2i', 'AATS3i', 'AATS4i', 'AATS5i', 'AATS6i', 'AATS7i', 'ATSC0c', 'ATSC1c', 'ATSC2c', 'ATSC3c', 'ATSC4c', 'ATSC5c', 'ATSC6c', 'ATSC7c', 'ATSC8c', 'ATSC0dv', 'ATSC1dv', 'ATSC2dv', 'ATSC3dv', 'ATSC4dv', 'ATSC5dv', 'ATSC6dv', 'ATSC7dv', 'ATSC8dv', 'ATSC0d', 'ATSC1d', 'ATSC2d', 'ATSC3d', 'ATSC4d', 'ATSC5d', 'ATSC6d', 'ATSC7d', 'ATSC8d', 'ATSC0s', 'ATSC1s', 'ATSC2s', 'ATSC3s', 'ATSC4s', 'ATSC5s', 'ATSC6s', 'ATSC7s', 'ATSC8s', 'ATSC0Z', 'ATSC1Z', 'ATSC2Z', 'ATSC3Z', 'ATSC4Z', 'ATSC5Z', 'ATSC6Z', 'ATSC7Z', 'ATSC8Z', 'ATSC0m', 'ATSC1m', 'ATSC2m', 'ATSC3m', 'ATSC4m', 'ATSC5m', 'ATSC6m', 'ATSC7m', 'ATSC8m', 'ATSC0v', 'ATSC1v', 'ATSC2v', 'ATSC3v', 'ATSC4v', 'ATSC5v', 'ATSC6v', 'ATSC7v', 'ATSC8v', 'ATSC0se', 'ATSC1se', 'ATSC2se', 'ATSC3se', 'ATSC4se', 'ATSC5se', 'ATSC6se', 'ATSC7se', 'ATSC8se', 'ATSC0pe', 'ATSC1pe', 'ATSC2pe', 'ATSC3pe', 'ATSC4pe', 'ATSC5pe', 'ATSC6pe', 'ATSC7pe', 'ATSC8pe', 'ATSC0are', 'ATSC1are', 'ATSC2are', 'ATSC3are', 'ATSC4are', 'ATSC5are', 'ATSC6are', 'ATSC7are', 'ATSC8are', 'ATSC0p', 'ATSC1p', 'ATSC2p', 'ATSC3p', 'ATSC4p', 'ATSC5p', 'ATSC6p', 'ATSC7p', 'ATSC8p', 'ATSC0i', 'ATSC1i', 'ATSC2i', 'ATSC3i', 'ATSC4i', 'ATSC5i', 'ATSC6i', 'ATSC7i', 'ATSC8i', 'AATSC0c', 'AATSC1c', 'AATSC2c', 'AATSC3c', 'AATSC4c', 'AATSC5c', 'AATSC6c', 'AATSC7c', 'AATSC0dv', 'AATSC1dv', 'AATSC2dv', 'AATSC3dv', 'AATSC4dv', 'AATSC5dv', 'AATSC6dv', 'AATSC7dv', 'AATSC0d', 'AATSC1d', 'AATSC2d', 'AATSC3d', 'AATSC4d', 'AATSC5d', 'AATSC6d', 'AATSC7d', 'AATSC0s', 'AATSC1s', 'AATSC2s', 'AATSC3s', 'AATSC4s', 'AATSC5s', 'AATSC6s', 'AATSC7s', 'AATSC0Z', 'AATSC1Z', 'AATSC2Z', 'AATSC3Z', 'AATSC4Z', 'AATSC5Z', 'AATSC6Z', 'AATSC7Z', 'AATSC0m', 'AATSC1m', 'AATSC2m', 'AATSC3m', 'AATSC4m', 'AATSC5m', 'AATSC6m', 'AATSC7m', 'AATSC0v', 'AATSC1v', 'AATSC2v', 'AATSC3v', 'AATSC4v', 'AATSC5v', 'AATSC6v', 'AATSC7v', 'AATSC0se', 'AATSC1se', 'AATSC2se', 'AATSC3se', 'AATSC4se', 'AATSC5se', 'AATSC6se', 'AATSC7se', 'AATSC0pe', 'AATSC1pe', 'AATSC2pe', 'AATSC3pe', 'AATSC4pe', 'AATSC5pe', 'AATSC6pe', 'AATSC7pe', 'AATSC0are', 'AATSC1are', 'AATSC2are', 'AATSC3are', 'AATSC4are', 'AATSC5are', 'AATSC6are', 'AATSC7are', 'AATSC0p', 'AATSC1p', 'AATSC2p', 'AATSC3p', 'AATSC4p', 'AATSC5p', 'AATSC6p', 'AATSC7p', 'AATSC0i', 'AATSC1i', 'AATSC2i', 'AATSC3i', 'AATSC4i', 'AATSC5i', 'AATSC6i', 'AATSC7i', 'MATS1c', 'MATS2c', 'MATS3c', 'MATS4c', 'MATS5c', 'MATS6c', 'MATS7c', 'MATS1dv', 'MATS2dv', 'MATS3dv', 'MATS4dv', 'MATS5dv', 'MATS6dv', 'MATS7dv', 'MATS1d', 'MATS2d', 'MATS3d', 'MATS4d', 'MATS5d', 'MATS6d', 'MATS7d', 'MATS1s', 'MATS2s', 'MATS3s', 'MATS4s', 'MATS5s', 'MATS6s', 'MATS7s', 'MATS1Z', 'MATS2Z', 'MATS3Z', 'MATS4Z', 'MATS5Z', 'MATS6Z', 'MATS7Z', 'MATS1m', 'MATS2m', 'MATS3m', 'MATS4m', 'MATS5m', 'MATS6m', 'MATS7m', 'MATS1v', 'MATS2v', 'MATS3v', 'MATS4v', 'MATS5v', 'MATS6v', 'MATS7v', 'MATS1se', 'MATS2se', 'MATS3se', 'MATS4se', 'MATS5se', 'MATS6se', 'MATS7se', 'MATS1pe', 'MATS2pe', 'MATS3pe', 'MATS4pe', 'MATS5pe', 'MATS6pe', 'MATS7pe', 'MATS1are', 'MATS2are', 'MATS3are', 'MATS4are', 'MATS5are', 'MATS6are', 'MATS7are', 'MATS1p', 'MATS2p', 'MATS3p', 'MATS4p', 'MATS5p', 'MATS6p', 'MATS7p', 'MATS1i', 'MATS2i', 'MATS3i', 'MATS4i', 'MATS5i', 'MATS6i', 'MATS7i', 'GATS1c', 'GATS2c', 'GATS3c', 'GATS4c', 'GATS5c', 'GATS6c', 'GATS7c', 'GATS1dv', 'GATS2dv', 'GATS3dv', 'GATS4dv', 'GATS5dv', 'GATS6dv', 'GATS7dv', 'GATS1d', 'GATS2d', 'GATS3d', 'GATS4d', 'GATS5d', 'GATS6d', 'GATS7d', 'GATS1s', 'GATS2s', 'GATS3s', 'GATS4s', 'GATS5s', 'GATS6s', 'GATS7s', 'GATS1Z', 'GATS2Z', 'GATS3Z', 'GATS4Z', 'GATS5Z', 'GATS6Z', 'GATS7Z', 'GATS1m', 'GATS2m', 'GATS3m', 'GATS4m', 'GATS5m', 'GATS6m', 'GATS7m', 'GATS1v', 'GATS2v', 'GATS3v', 'GATS4v', 'GATS5v', 'GATS6v', 'GATS7v', 'GATS1se', 'GATS2se', 'GATS3se', 'GATS4se', 'GATS5se', 'GATS6se', 'GATS7se', 'GATS1pe', 'GATS2pe', 'GATS3pe', 'GATS4pe', 'GATS5pe', 'GATS6pe', 'GATS7pe', 'GATS1are', 'GATS2are', 'GATS3are', 'GATS4are', 'GATS5are', 'GATS6are', 'GATS7are', 'GATS1p', 'GATS2p', 'GATS3p', 'GATS4p', 'GATS5p', 'GATS6p', 'GATS7p', 'GATS1i', 'GATS2i', 'GATS3i', 'GATS4i', 'GATS5i', 'GATS6i', 'GATS7i']
                df = df[common_columns]
                dataframes[name] = df
            # Calculate the minimum count of non-null values required 
            min_count = int(((100 - threshold) / 100) * df.shape[0] + 1)
            # Drop columns with insufficient non-null values
            df_cleaned = df.dropna(axis=1, thresh=min_count)
            dataframes[name] = df_cleaned
        self.dataframes = dataframes

    def ImputeData(self, method="knn", class_specific = False):
        """

        Impute missing data in the dataframes.

        This method supports five types of imputation methods: 
           - 'knn' will use the K-Nearest Neighbors approach to impute missing values.
           - 'mean' will use the mean of each column to fill missing values.
           - 'most_frequent' will use the mode of each column to fill missing values.
           - 'median' will use the median of each column to fill missing values.
           - 'interpolate' will use the Interpolation method to fill missing values.

        :param method: The imputation method to use. Options are "knn", "mean", "mode", "median", and "interpolate".
        :type method: str, optional

        """
       
        dataframes = self.dataframes
        for name, df in dataframes.items():
            missing_values = df.isnull().sum().sum()
            if missing_values > 0:
                if method == "knn":
                    if class_specific is True:
                        df_imputed = impute_class_specific(df, self.prediction_label)
                    else:
                        imputer = KNNImputer(n_neighbors=5)
                        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
                elif method in ["mean", "most_frequent", "median"]:
                    imputer = SimpleImputer(strategy=method)
                    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
                elif method == "interpolate":
                    df.interpolate(method='linear', inplace=True)
                    df_imputed = df
                else:
                    raise ValueError("Please select a valid imputation method: 'knn', 'mean', 'most_frequent', 'median', or 'interpolate'")
                dataframes[name] = df_imputed
        self.dataframes = dataframes
        print("Imputation Done")


    def scale_data(self, scaling_type,  **kwargs):
        """

        Scale the dataFrame based on the specified scale type.

        The scaling methods are as follows:
            - 'standardize' Applies the standard scaler method to each column of the dataframe. This method transforms the data to have a mean of zero and a standard deviation of one. It is useful for reducing the effect of outliers and making the data more normally distributed.
            - 'minmax' Applies the min-max scaler method to each column of the dataframe. This method transforms the data to have a minimum value of zero and a maximum value of one. It is useful for making the data more comparable and preserving the original distribution.
            - 'robust' Applies the robust scaler method to each column of the dataframe. This method transforms the data using the median and the interquartile range. It is useful for reducing the effect of outliers and making the data more robust to noise.
            - 'pareto' Applies the pareto scaling method to each column of the dataframe. This method divides each element by the square root of the standard deviation of the column. It is useful for making the data more homogeneous and reducing the effect of skewness.
        
        :param scaling_type: The type of scaling to be applied. It can be one of these: 'standardize', 'minmax', 'robust', or 'pareto'.
        :type scaling_type: str
        :param kwargs: Additional parameters for specific scaling methods.
        :type kwargs: dict


        """

        dataframes = self.dataframes


        # Loop through the df_dict
        for dataset_name, df in dataframes.items():

            # Apply the scaling type to the dataframe
            if scaling_type == 'standardize':
                scaler = StandardScaler()
                scaled_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
            elif scaling_type == 'minmax':
                scaler = MinMaxScaler()
                scaled_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
            elif scaling_type == 'robust':
                scaler = RobustScaler()
                scaled_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
            elif scaling_type == 'pareto':
                scaled_df = df / np.sqrt(df.std())
            else:
                raise ValueError(f"""Unsupported scaling type: {scaling_type} 
                                    possible values are 'standardize', 'minmax', 'robust', or 'pareto' """)
            # Return the scaled dataframe
            dataframes[dataset_name] = scaled_df
        self.dataframes = dataframes

    def normalize_data(self,normalization_type, **kwargs):
        """

        This method supports four types of normalization methods:
            - 'constant sum' normalization. The default sum is 1, which can  specified using the 'sum' keyword argument in kwargs. It is a method used to normalize data such that the sum of values for each observation remains constant. It ensures that the relative contributions of individual features to the total sum are compared rather than the absolute values. This normalization technique is beneficial for comparing samples with different total magnitudes but similar distributions. Mathematically, each observation is normalized by dividing it by the sum of its values, and then multiplying by a constant factor to achieve the desired sum.
            - 'L1' normalization (Lasso Norm or Manhattan Norm): Also known as Lasso Norm or Manhattan Norm. Each observation vector is rescaled by dividing each element by the L1-norm of the vector.. L1-norm of a vector is the sum of the absolute values of its components. Mathematically, for a vector x, L1 normalization is given by: L1-Norm = ∑|x_i|. After L1 normalization, the sum of the absolute values of the elements in each vector becomes 1. Widely used in machine learning tasks such as Lasso regression to encourage sparsity in the solution.
            - 'L2' normalization (Ridge Norm or Euclidean Norm): Also known as Ridge Norm or Euclidean Norm. Each observation vector is rescaled by dividing each element by the L2-norm of the vector. L2-norm of a vector is the square root of the sum of the squares of its components. Mathematically, for a vector x, L2 normalization is given by: L2-Norm = √∑x_i^2. After L2 normalization, the Euclidean distance (or the magnitude) of each vector becomes 1. Widely used in various machine learning algorithms such as logistic regression, support vector machines, and neural networks.
            - 'max' normalization (Maximum Normalization): Scales each feature in the dataset by dividing it by the maximum absolute value of that feature across all observations. Ensures that each feature's values are within the range [-1, 1] or [0, 1] depending on whether negative values are present or not. Useful when the ranges of features in the dataset are significantly different, preventing certain features from dominating the learning process due to their larger scales. Commonly used in neural networks and deep learning models as part of the data preprocessing step.

        Normalize dataframes using different types of normalization.

        :param normalization_type: The type of normalization to apply. It can be one of these: 'constant_sum', 'L1', 'L2', or 'max'.        
        :type normalization_type: str
        :param kwargs: Additional arguments for some normalization types.
        :type kwargs: dict
        :raises ValueError: If the provided method is not 'constant_sum', 'L1' ,'L2' or 'max

        """


        # Create an empty dictionary to store the normalized dataframes
        normalized_df_dict = {}
        dataframes = self.dataframes

        # Loop through the df_dict
        for dataset_name, df in dataframes.items():

            # Apply the normalization type to the dataframe
            if normalization_type == 'constant_sum':
                constant_sum = kwargs.get('sum', 1)
                axis = kwargs.get('axis', 1)
                normalized_df = normalize_to_constant_sum(df, constant_sum=constant_sum, axis=axis)
            elif normalization_type == 'L1':
                normalized_df =  pd.DataFrame(normalize(df, norm='l1'), columns=df.columns,index = df.index)
            elif normalization_type == 'L2':
                normalized_df =  pd.DataFrame(normalize(df, norm='l2'), columns=df.columns,index = df.index)
            elif normalization_type == 'max':
                normalized_df = pd.DataFrame(normalize(df, norm='max'), index=df.index )
            else:
                raise ValueError(f"Unsupported normalization type: {normalization_type} \n possible values are 'constant_sum', 'L1' ,'L2' and 'max'   ")
            # Store the normalized dataframe in the normalized_df_dict
            dataframes[dataset_name] = normalized_df
        self.dataframes = dataframes

    def transform_data(self, transformation_type, **kwargs):
        """
        
        The transformation methods are as follows:
            - 'cubicroot': Applies the cube root function to each element of the dataframe.
            - 'log10': Applies the base 10 logarithm function to each element of the dataframe.
            - 'log': Applies the natural logarithm function to each element of the dataframe.
            - 'log2': Applies the base 2 logarithm function to each element of the dataframe.
            - 'sqrt': Applies the square root function to each element of the dataframe.
            - 'powertransformer': Applies the power transformer method to each column of the dataframe. This method transforms the data to make it more Gaussian-like. It supports two methods: 'yeo-johnson' and 'box-cox'. The default method is 'yeo-johnson', which can handle both positive and negative values. The 'box-cox' method can only handle positive values. The method can be specified using the 'method' keyword argument in kwargs.
            - 'quantiletransformer': Applies the quantile transformer method to each column of the dataframe. This method transforms the data to follow a uniform or a normal distribution. It supports two output distributions: 'uniform' and 'normal'. The default distribution is 'uniform'. The distribution can be specified using the 'output_distribution' keyword argument in kwargs.

        Transform dataframes using different types of mathematical transformations.

        :param transformation_type: The type of transformation to apply. It can be one of these: 'cubicroot', 'log10', 'log', 'log2', 'sqrt', 'powertransformer', or 'quantiletransformer'.
        :type transformation_type: str
        :param kwargs: Additional arguments for some transformation types.
        :type kwargs: dict


        :raises: ValueError if the transformation_type is not one of the valid options.

        """

        dataframes = self.dataframes

        # Loop through the df_dict
        for dataset_name, df in dataframes.items():

            # Apply the normalization type to the dataframe
            # Apply the transformation type to the dataframe
            if transformation_type == 'cubicroot':
                transformed_df = np.cbrt(df)
            elif transformation_type == 'log10':
                transformed_df = np.log10(df)
            elif transformation_type == 'log':
                transformed_df = np.log(df)
            elif transformation_type == 'log2':
                transformed_df = np.log2(df)
            elif transformation_type == 'sqrt':
                transformed_df = np.sqrt(df)
            elif transformation_type == 'powertransformer':
                # Create a scaler object with the specified method
                method = kwargs.get('method', 'yeo-johnson')
                scaler = PowerTransformer(method=method)

                # Transform the dataframe and convert the result to a dataframe
                transformed_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
            elif transformation_type == 'quantiletransformer':
                # Create a scaler object with the specified output distribution
                output_distribution = kwargs.get('output_distribution', 'uniform')
                scaler = QuantileTransformer(output_distribution=output_distribution)

                # Transform the dataframe and convert the result to a dataframe
                transformed_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
            else:
                raise ValueError(f"Unsupported transformation type: {transformation_type} possible values are \n 'cubicroot', 'log10', 'log', 'log2', 'sqrt', 'powertransformer', or 'quantiletransformer'")
            # Return the transformed dataframe
            
            dataframes[dataset_name] = transformed_df
        self.dataframes = dataframes




    def fuseFeatures(self, n_components, methods=["pca","AER"],AER_dim=4096,save_dir = "ChemicalDice_fusedData",**kwargs):
        """

        The fusion methods are as follows:
           - 'AER': Autoencoder Reconstruction Analysis, a autoencoder based feature fusion technique that cross reconstruct data from different modalities and make autoencoders to learn importent features from the data.  Finally the data is converted from its orignal dimention to reduced size of embedding  
           - 'pca': Principal Component Analysis, a linear dimensionality reduction technique that projects the data onto a lower-dimensional subspace that maximizes the variance.
           - 'ica': Independent Component Analysis, a linear dimensionality reduction technique that separates the data into independent sources based on the assumption of statistical independence.
           - 'ipca': Incremental Principal Component Analysis, a variant of PCA that allows for online updates of the components without requiring access to the entire dataset.
           - 'cca': Canonical Correlation Analysis, a linear dimensionality reduction technique that finds the linear combinations of two sets of variables that are maximally correlated with each other.
           - 'tsne': t-distributed Stochastic Neighbor Embedding, a non-linear dimensionality reduction technique that preserves the local structure of the data by embedding it in a lower-dimensional space with a probability distribution that matches the pairwise distances in the original space.
           - 'kpca': Kernel Principal Component Analysis, a non-linear extension of PCA that uses a kernel function to map the data into a higher-dimensional feature space where it is linearly separable.
           - 'rks': Random Kitchen Sinks, a non-linear dimensionality reduction technique that uses random projections to approximate a kernel function and map the data into a lower-dimensional feature space.
           - 'SEM': Structural Equation Modeling, a statistical technique that tests the causal relationships between multiple variables using a combination of regression and factor analysis.
           - 'isomap': Isometric Mapping, a non-linear dimensionality reduction technique that preserves the global structure of the data by embedding it in a lower-dimensional space with a geodesic distance that approximates the shortest path between points in the original space.
           - 'lle': Locally Linear Embedding, a non-linear dimensionality reduction technique that preserves the local structure of the data by reconstructing each point as a linear combination of its neighbors and embedding it in a lower-dimensional space that minimizes the reconstruction error.
           - 'autoencoder': A type of neural network that learns to encode the input data into a lower-dimensional latent representation and decode it back to the original data, minimizing the reconstruction error.
           - 'plsda': Partial Least Squares Discriminant Analysis, a supervised dimensionality reduction technique that finds the linear combinations of the features that best explain both the variance and the correlation with the target variable.
           - 'tensordecompose': Tensor Decomposition, a technique that decomposes a multi-dimensional array (tensor) into a set of lower-dimensional arrays (factors) that capture the latent structure and interactions of the data.
        
        Fuse the features of multiple dataframes using different methods.

        :param n_components: The number of components use for the fusion.
        :type n_components: int
        :param method: The method to use for the fusion. It can be one of these: 'pca', 'ica', 'ipca', 'cca', 'tsne', 'kpca', 'rks', 'SEM', 'isomap', 'lle', 'autoencoder', 'plsda', or 'tensordecompose'.
        :type methods: list
        :param kwargs: Additional arguments for specific fusion methods.
        :type kwargs: dict

        :raises: ValueError if the method is not one of the valid options.
        """
        
        try:
            os.mkdir(save_dir)
        except:
            print(save_dir," already exists")

        methods_chemdices = ['pca', 'ica', 'ipca', 'cca', 'tsne', 'kpca', 'rks', 'SEM', 'autoencoder', 'tensordecompose', 'plsda',"AER"]   
        valid_methods_chemdices = [method for method in methods if method in methods_chemdices]
        invalid_methods_chemdices = [method for method in methods if method not in methods_chemdices]
        methods_chemdices_text = ",".join(methods_chemdices)
        invalid_methods_chemdices_text = ",".join(invalid_methods_chemdices)
        if len(invalid_methods_chemdices):
            ValueError(f"These methods are invalid:{invalid_methods_chemdices_text}\n Valid methods are : {methods_chemdices_text}")
        dataframe = self.dataframes
        for method in valid_methods_chemdices:
            print(method)
            # Iterate through the dictionary and fusion DataFrame
            # train data fusion
            df_list = []
            for name, df in dataframe.items():
                df_list.append(df)
            if method in ['pca', 'ica', 'ipca']:
                merged_df = pd.concat(df_list, axis=1)
                fused_df1 = apply_analysis_linear1(merged_df, analysis_type=method, n_components=n_components, **kwargs)
                fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(n_components)+".csv"))
            elif method in ['cca']:
                all_combinations = []
                all_combinations.extend(combinations(df_list, 2))
                all_fused =[]
                n=0
                for df_listn in all_combinations:
                    fused_df_t = ccafuse(df_listn[0], df_listn[1],n_components)
                    all_fused.append(fused_df_t)
                fused_df1 = pd.concat(all_fused, axis=1, sort=False)
                fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(n_components)+".csv"))

            elif method in ['tsne', 'kpca', 'rks', 'SEM']:
                merged_df = pd.concat(df_list, axis=1)
                fused_df1 = apply_analysis_nonlinear1(merged_df,
                                                    analysis_type=method,
                                                    n_components=n_components,
                                                    **kwargs)
                fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(n_components)+".csv"))

            elif method in ['isomap', 'lle']:
                merged_df = pd.concat(df_list, axis=1)
                fused_df1 = apply_analysis_nonlinear2(merged_df, analysis_type=method, n_neighbors=5, n_components=n_components, **kwargs)
                fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(n_components)+".csv"))

            elif method in ['autoencoder']:
                merged_df = pd.concat(df_list, axis=1)
                fused_df1 = apply_analysis_nonlinear3(merged_df,
                                                    analysis_type=method,
                                                    lr=0.001,
                                                    num_epochs = 20, 
                                                    hidden_sizes=[128, 64, 36, 18],
                                                    **kwargs)
                fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(n_components)+".csv"))

            elif method in ['AER']:
                scaler =StandardScaler()
                if self.training_AER_model is None or self.AER_model_embt_size != AER_dim:
                    df_list2=[None,None,None,None,None,None]
                    for name, df in self.dataframes.items():
                        if name.lower() == "mopac":
                            df_list2[0] = df.copy()
                        elif name.lower() == "chemberta":
                            df_list2[1] = df.copy()
                        elif name.lower() ==  "mordred":
                            df_list2[2] = df.copy()
                        elif name.lower() ==  "signaturizer":
                            df_list2[3] = df.copy()
                        elif name.lower() ==  "imagemol":
                            df_list2[4] = df.copy()
                        elif name.lower() ==  "grover":
                            df_list2[5] = df.copy()
                    if type(AER_dim) == list:
                        embd = 8192
                        #print(df_list2)
                        embeddings_8192 = AutoencoderReconstructor_training_8192(df_list2[0], df_list2[1], df_list2[2], df_list2[3],df_list2[4],df_list2[5])
                        fused_df_unstandardized = embeddings_8192
                        fused_df_unstandardized.set_index("id",inplace =True)
                        fused_df1 = pd.DataFrame(scaler.fit_transform(fused_df_unstandardized), index=fused_df_unstandardized.index, columns=fused_df_unstandardized.columns)
                        if 8192 in AER_dim:
                            fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(embd)+".csv"))
                            AER_dim.remove(8192)
                        for embd in AER_dim:
                            embeddings_df = AutoencoderReconstructor_training_other(df_list2[0], df_list2[1], df_list2[2], df_list2[3],df_list2[4],df_list2[5], embd)
                            fused_df_unstandardized = embeddings_df
                            fused_df_unstandardized.set_index("id",inplace =True)
                            fused_df1 = pd.DataFrame(scaler.fit_transform(fused_df_unstandardized), index=fused_df_unstandardized.index, columns=fused_df_unstandardized.columns)
                            fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(embd)+".csv"))
                    elif type(AER_dim) == int:
                        embeddings_df = AutoencoderReconstructor_training_single(df_list2[0], df_list2[1], df_list2[2], df_list2[3],df_list2[4],df_list2[5],AER_dim)
                        fused_df_unstandardized = embeddings_df
                        fused_df_unstandardized.set_index("id",inplace =True)
                        fused_df1 = pd.DataFrame(scaler.fit_transform(fused_df_unstandardized), index=fused_df_unstandardized.index, columns=fused_df_unstandardized.columns)
                        fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(embd)+".csv"))
                    else:
                        ValueError("AER_dim should be  int or list")

                    # explanibility = {}
                    # for name, df in self.dataframes.items():
                    #     if name.lower() == "mopac":
                    #         column_name = df.columns
                    #         explanibility['name'] = pd.DataFrame({'Feature':column_name,'weights':model_wt[0]})
                    #     elif name.lower() == "chemberta":
                    #         column_name = df.columns
                    #         explanibility['name'] = pd.DataFrame({'Feature':column_name,'weights':model_wt[1]})
                    #     elif name.lower() ==  "mordred":
                    #         column_name = df.columns
                    #         explanibility['name'] = pd.DataFrame({'Feature':column_name,'weights':model_wt[2]})
                    #     elif name.lower() ==  "signaturizer":
                    #         column_name = df.columns
                    #         explanibility['name'] = pd.DataFrame({'Feature':column_name,'weights':model_wt[3]})
                    #     elif name.lower() ==  "imagemol":
                    #         column_name = df.columns
                    #         explanibility['name'] = pd.DataFrame({'Feature':column_name,'weights':model_wt[4]})
                    #     elif name.lower() ==  "grover":
                    #         column_name = df.columns
                    #         explanibility['name'] = pd.DataFrame({'Feature':column_name,'weights':model_wt[5]})
                    # self.training_AER_model = model_state
                    # self.AER_model_embt_size = embd
                    # self.AER_model_explainability = explanibility
                else:
                    print("AER model Training")
                # embd = AER_dim
                # df_list2=[None,None,None,None,None,None]
                # for name, df in dataframe.items():
                #     if name.lower() == "mopac":
                #         df_list2[0] = df.copy()
                #     elif name.lower() == "chemberta":
                #         df_list2[1] = df.copy()
                #     elif name.lower() ==  "mordred":
                #         df_list2[2] = df.copy()
                #     elif name.lower() ==  "signaturizer":
                #         df_list2[3] = df.copy()
                #     elif name.lower() ==  "imagemol":
                #         df_list2[4] = df.copy()
                #     elif name.lower() ==  "grover":
                #         df_list2[5] = df.copy()
                
                # model_state = self.training_AER_model
                # all_embeddings = AutoencoderReconstructor_testing(df_list2[0],df_list2[1],df_list2[2],df_list2[3],df_list2[4],df_list2[5],embedding_sizes=embd, model_state=model_state)
                
                # scaler = StandardScaler()
                # if len(all_embeddings) > 1:
                #     fused_df1 = []
                #     for i in range(len(all_embeddings)):
                #         fused_df_unstandardized = all_embeddings[i]
                #         fused_df_unstandardized.set_index("id",inplace =True)
                #         fused_df = pd.DataFrame(scaler.fit_transform(fused_df_unstandardized), index=fused_df_unstandardized.index, columns=fused_df_unstandardized.columns)
                #         fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+all_embeddings[i]+".csv"))
                # else:
                #     fused_df_unstandardized = all_embeddings[0]
                #     fused_df_unstandardized.set_index("id",inplace =True)
                #     fused_df1 = pd.DataFrame(scaler.fit_transform(fused_df_unstandardized), index=fused_df_unstandardized.index, columns=fused_df_unstandardized.columns)
                #     fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+all_embeddings[0]+".csv"))
                    
            prediction_label = self.prediction_label
            if method in ['plsda']:
                df_list = []
                for name, df in dataframe.items():
                    df_list.append(df)
                merged_df = pd.concat(df_list, axis=1)
                fused_df1, pls_canonical = apply_analysis_linear2(merged_df, prediction_label, analysis_type=method, n_components=n_components, **kwargs)
                self.pls_model = pls_canonical
                fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(n_components)+".csv"))

            ######################
                
            if method in ['tensordecompose']:
                df_list = []
                for name, df in dataframe.items():
                    df_list.append(df)
                df_list_selected=[]
                top_features = []
                for df in df_list:
                    num_features = 100
                    fs = SelectKBest(score_func=f_regression, k=num_features)
                    #print(df)
                    X_selected = fs.fit_transform(df, prediction_label)
                    # print(fs.get_feature_names_out())
                    #print(df.columns)
                    top_feature = list(fs.get_feature_names_out())
                    top_features.append(top_feature)
                    df_list_selected.append(df[top_feature])
                all_selected = np.array(df_list_selected)
                fused_df = apply_analysis_nonlinear4(all_selected,
                                                    analysis_type=method,
                                                    n_components=n_components,
                                                    tol=10e-6,
                                                    **kwargs)
                fused_df1 = pd.DataFrame(fused_df, index =df.index, columns = [f'TD{i+1}' for i in range(fused_df.shape[1])])
                self.top_features = top_features
                fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(n_components)+".csv"))
            print("Data is fused and saved to  ChemicalDice_fusedData")


    def evaluate_fusion_models_nfold(self, folds, task_type, fused_data_path="ChemicalDice_fusedData", models = None):
        """
        Perform n-fold cross-validation on fusion models and save the evaluation metrics.This method evaluates the performance of various machine learning models on fused data obtained from ChemDice. It supports both classification and regression tasks and saves the performance metrics for each fold into a CSV file.
        :param folds: The number of folds to use for KFold cross-validation.
        :type folds: int
        :param task_type: The type of task to perform, either 'classification' or 'regression'.
        :type task_type: str
        :param fused_data_path: The path to the directory containing the fused data files, defaults to 'ChemicalDice_fusedData'.
        :type fused_data_path: str
        :param models: The list of model names to evaluate. If None, a default set of models will be used.
        :type models: list[str], optional
        :raises ValueError: If the `task_type` is neither 'classification' nor 'regression'.
        :return: None

        Available Models for
        Classification:
            - "Logistic Regression"
            - "Decision Tree"
            - "Random Forest"
            - "Support Vector Machine"
            - "Naive Bayes"
            - "KNN"
            - "NeuralNet"
            - "QDA"
            - "AdaBoost"
            - "Extra Trees"
            - "XGBoost"

        Available Models for
        Regression:
            - "Linear Regression"
            - "Ridge"
            - "Lasso"
            - "ElasticNet"
            - "Decision Tree"
            - "Random Forest"
            - "Gradient Boosting"
            - "AdaBoost"
            - "Support Vector Machine"
            - "K Neighbors"
            - "MLP"
            - "Gaussian Process"
            - "Kernel Ridge"

        .. note:: The method assumes that the prediction labels are stored in `self.prediction_label`.For classification, it evaluates models based on AUC, accuracy, precision, recall, f1 score, balanced accuracy, MCC, and kappa. For regression, it evaluates models based on R2 score, MSE, RMSE, and MAE. The results are saved in a CSV file named 'Accuracy_Metrics_{method_name}.csv' in a directory named '{folds}_fold_CV_results'.

        """
        
        list_of_files = os.listdir(fused_data_path)
        self.task_type = task_type
        self.folds = folds
        try:
            os.mkdir(f"{folds}_fold_CV_results")
        except:
            print(f"{folds}_fold_CV_results already exists")

        for file in list_of_files:
            y = self.prediction_label
            kf = KFold(n_splits=folds, shuffle=True, random_state=42)

            if task_type == "classification":
                if models == None:
                    models = [
                        ("Logistic Regression", LogisticRegression()),
                        ("Decision Tree", DecisionTreeClassifier()),
                        ("Random Forest", RandomForestClassifier()),
                        ("Support Vector Machine", SVC(probability=True)),
                        ("Naive Bayes", GaussianNB()),
                        ("KNN", KNeighborsClassifier(leaf_size=1, n_neighbors=11, p=3, weights='distance')),
                        ("NeuralNet", MLPClassifier(alpha=1, max_iter=1000)),
                        ("QDA", QuadraticDiscriminantAnalysis()),
                        ("AdaBoost", AdaBoostClassifier()),
                        ("Extra Trees", ExtraTreesClassifier()),
                        ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
                    ]
                else:
                    classifiers = [
                        ("Logistic Regression", LogisticRegression()),
                        ("Decision Tree", DecisionTreeClassifier()),
                        ("Random Forest", RandomForestClassifier()),
                        ("Support Vector Machine", SVC(probability=True)),
                        ("Naive Bayes", GaussianNB()),
                        ("KNN", KNeighborsClassifier(leaf_size=1, n_neighbors=11, p=3, weights='distance')),
                        ("NeuralNet", MLPClassifier(alpha=1, max_iter=1000)),
                        ("QDA", QuadraticDiscriminantAnalysis()),
                        ("AdaBoost", AdaBoostClassifier()),
                        ("Extra Trees", ExtraTreesClassifier()),
                        ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
                    ]
                    models = [clf for clf in classifiers if clf[0] in models]

                metrics = {
                    "Model type": [],
                    "Fold": [],
                    "Model": [],
                    "AUC": [],
                    "Accuracy": [],
                    "Precision": [],
                    "Recall": [],
                    "f1 score": [],
                    "Balanced accuracy": [],
                    "MCC": [],
                    "Kappa": []
                }
            elif task_type == "regression":
                if models == None:
                    models = [
                        ("Linear Regression", LinearRegression()),
                        ("Ridge", Ridge()),
                        ("Lasso", Lasso()),
                        ("ElasticNet", ElasticNet()),
                        ("Decision Tree", DecisionTreeRegressor()),
                        ("Random Forest", RandomForestRegressor()),
                        ("Gradient Boosting", GradientBoostingRegressor()),
                        ("AdaBoost", AdaBoostRegressor()),
                        ("Support Vector Machine", SVR()),
                        ("K Neighbors", KNeighborsRegressor()),
                        ("MLP", MLPRegressor()),
                        ("Gaussian Process", GaussianProcessRegressor()),
                        ("Kernel Ridge", KernelRidge())
                    ]
                else:
                    regressors = [
                        ("Linear Regression", LinearRegression()),
                        ("Ridge", Ridge()),
                        ("Lasso", Lasso()),
                        ("ElasticNet", ElasticNet()),
                        ("Decision Tree", DecisionTreeRegressor()),
                        ("Random Forest", RandomForestRegressor()),
                        ("Gradient Boosting", GradientBoostingRegressor()),
                        ("AdaBoost", AdaBoostRegressor()),
                        ("Support Vector Machine", SVR()),
                        ("K Neighbors", KNeighborsRegressor()),
                        ("MLP", MLPRegressor()),
                        ("Gaussian Process", GaussianProcessRegressor()),
                        ("Kernel Ridge", KernelRidge())
                    ]

                    models = [regg for regg in regressors if regg[0] in models]

                metrics = {
                    "Model type": [],
                    "Model": [],
                    "Fold": [],
                    "R2 Score": [],
                    "MSE": [],
                    "RMSE": [],
                    "MAE": []
                }
            else:
                raise ValueError("task_type can be either 'classification' or 'regression'")

            fold_number = 0

            for train_index, test_index in kf.split(y):
                fold_number += 1
                if file.startswith("fused_data"):
                    method_chemdice = file.replace("fused_data_", "")
                    method_chemdice= method_chemdice.replace(".csv","")
                    # print(method_chemdice)
                    if method_chemdice.startswith("plsda"):
                        print("Method name",method_chemdice)
                        n_components = method_chemdice.replace("plsda_","")
                        n_components = int(n_components.replace(".csv",""))
                        train_dataframes, test_dataframes, train_label, test_label  = save_train_test_data_n_fold(self.dataframes, self.prediction_label, train_index, test_index, output_dir="comaprision_data_fold_"+str(fold_number)+"_of_"+str(fold_number))
                        X_train = self.fuseFeaturesTrain_plsda(n_components = n_components,  method=method_chemdice, train_dataframes = train_dataframes, train_label =train_label)
                        y_train = train_label
                        X_test = self.fuseFeaturesTest_plsda(n_components = n_components,  method=method_chemdice,test_dataframes = test_dataframes)
                        y_test = test_label
                        # print("y_test")
                        # print(y_test)
                    elif method_chemdice.startswith("tensordecompose"):
                        print("Method name",method_chemdice)
                        n_components = method_chemdice.replace("tensordecompose_","")
                        n_components = int(n_components.replace(".csv",""))
                        train_dataframes, test_dataframes, train_label, test_label  = save_train_test_data_n_fold(self.dataframes, self.prediction_label, train_index, test_index, output_dir="comaprision_data_fold_"+str(fold_number)+"_of_"+str(fold_number))
                        X_train = self.fuseFeaturesTrain_td(n_components = n_components,  method=method_chemdice, train_dataframes = train_dataframes, train_label =train_label)
                        y_train = train_label
                        X_test = self.fuseFeaturesTest_td(n_components = n_components,  method=method_chemdice,test_dataframes = test_dataframes)
                        y_test = test_label
                        # print("y_test")
                        # print(y_test)
                    
                    else:
                        data = pd.read_csv(os.path.join(fused_data_path, file),index_col=0)
                        X = data
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y[train_index], y[test_index]

                    for name, model in models:


                        if method_chemdice.startswith(('pca', 'ica', 'ipca', 'cca', 'plsda')):
                            metrics["Model type"].append("linear")
                        else:
                            metrics["Model type"].append("Non-linear")
                        

                        name = f"{method_chemdice} {name}"
                        # print(X_train)
                        # print(y_train)
                        # Fit the model on the train set
                        model.fit(X_train, y_train)

                        if task_type == "classification":
                            # Predict the probabilities on the test set
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                            # Compute the metrics
                            auc = roc_auc_score(y_test, y_pred_proba)
                            accuracy = accuracy_score(y_test, y_pred_proba > 0.5)
                            precision = precision_score(y_test, y_pred_proba > 0.5)
                            recall = recall_score(y_test, y_pred_proba > 0.5)
                            f1 = f1_score(y_test, y_pred_proba > 0.5)
                            baccuracy = balanced_accuracy_score(y_test, y_pred_proba > 0.5)
                            mcc = matthews_corrcoef(y_test, y_pred_proba > 0.5)
                            kappa = cohen_kappa_score(y_test, y_pred_proba > 0.5)
                            # Append the metrics to the dictionary
                            metrics["Model"].append(name)
                            metrics["Fold"].append(fold_number)
                            metrics["AUC"].append(auc)
                            metrics["Accuracy"].append(accuracy)
                            metrics["Precision"].append(precision)
                            metrics["Recall"].append(recall)
                            metrics["f1 score"].append(f1)
                            metrics["Balanced accuracy"].append(baccuracy)
                            metrics["MCC"].append(mcc)
                            metrics["Kappa"].append(kappa)
                        else:
                            y_pred = model.predict(X_test)
                            # Compute the metrics
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            # Append the metrics to the dictionary
                            metrics["Model"].append(name)
                            metrics["Fold"].append(fold_number)
                            metrics["MSE"].append(mse)
                            metrics["RMSE"].append(rmse)
                            metrics["MAE"].append(mae)
                            metrics["R2 Score"].append(r2)
            #print(metrics)
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(f"{folds}_fold_CV_results/Accuracy_Metrics_{method_chemdice}.csv", index=False)
            print("Done")
            print(f"{folds}_fold_CV_results/Accuracy_Metrics_{method_chemdice}.csv saved")

    

    def evaluate_fusion_models_scaffold_split(self, split_type, task_type, fused_data_path="ChemicalDice_fusedData",models=None):
        """
        Perform n-fold cross-validation on fusion models and save the evaluation metrics. This method evaluates the performance of various machine learning models on fused data obtained from ChemDice. It supports both classification and regression tasks and saves the performance metrics for each fold into a CSV file.
        
        :param split_type: The type scaffold dsta split to perform. Three types available  'random', 'balanced' or 'simple'
        :type folds: str
        :param task_type: The type of task to perform, either 'classification' or 'regression'.
        :type task_type: str
        :param fused_data_path: The path to the directory containing the fused data files, defaults to 'ChemicalDice_fusedData'.
        :type fused_data_path: str
        :param models: The list of model names to evaluate. If None, a default set of models will be used.
        :type models: list[str], optional
        :raises ValueError: If the `task_type` is neither 'classification' nor 'regression'.
        :return: None

        Available Models for
        Classification:
            - "Logistic Regression"
            - "Decision Tree"
            - "Random Forest"
            - "Support Vector Machine"
            - "Naive Bayes"
            - "KNN"
            - "NeuralNet"
            - "QDA"
            - "AdaBoost"
            - "Extra Trees"
            - "XGBoost"

        Available Models for
        Regression:
            - "Linear Regression"
            - "Ridge"
            - "Lasso"
            - "ElasticNet"
            - "Decision Tree"
            - "Random Forest"
            - "Gradient Boosting"
            - "AdaBoost"
            - "Support Vector Machine"
            - "K Neighbors"
            - "MLP"
            - "Gaussian Process"
            - "Kernel Ridge"
            
        .. note:: The method assumes that the prediction labels are stored in `self.prediction_label`. For classification, it evaluates models based on AUC, accuracy, precision, recall, f1 score, balanced accuracy, MCC, and kappa. For regression, it evaluates models based on R2 score, MSE, RMSE, and MAE. The results are saved in a CSV file named 'Accuracy_Metrics_{method_chemdice}.csv' in a directory named 'scaffold_split_results'.


        """
        list_of_files = os.listdir(fused_data_path)
        self.task_type = task_type
        try:
            os.mkdir("scaffold_split_results")
        except:
            print("scaffold_split_results already exist")

        for file in list_of_files:
            method_chemdice = file.replace("fused_data_", "")
            method_chemdice= method_chemdice.replace(".csv","")
            print(method_chemdice)
            y = self.prediction_label

            if task_type == "classification":
                test_metrics = {
                    "Model": [],
                    "Model type":[],
                    "AUC": [],
                    "Accuracy": [],
                    "Precision": [],
                    "Recall": [],
                    "f1 score":[],
                    "Balanced accuracy":[],
                    "MCC":[],
                    "Kappa":[],
                }

                train_metrics = {
                    "Model": [],
                    "Model type":[],
                    "AUC": [],
                    "Accuracy": [],
                    "Precision": [],
                    "Recall": [],
                    "f1 score":[],
                    "Balanced accuracy":[],
                    "MCC":[],
                    "Kappa":[],
                }


                val_metrics = {
                    "Model": [],
                    "Model type":[],
                    "AUC": [],
                    "Accuracy": [],
                    "Precision": [],
                    "Recall": [],
                    "f1 score":[],
                    "Balanced accuracy":[],
                    "MCC":[],
                    "Kappa":[],
                }
            elif task_type == "regression":
                test_metrics = {
                    "Model": [],
                    "Model type":[],
                    "R2 Score": [],
                    "MSE": [],
                    "RMSE":[],
                    "MAE":[]
                    }
                
                train_metrics = {
                    "Model": [],
                    "Model type":[],
                    "R2 Score": [],
                    "MSE": [],
                    "RMSE":[],
                    "MAE":[]
                    }

                val_metrics = {
                    "Model": [],
                    "Model type":[],
                    "R2 Score": [],
                    "MSE": [],
                    "RMSE":[],
                    "MAE":[]
                    }
            else:
                raise ValueError("task_type can be either 'classification' or 'regression'")

            if split_type == "random":
                train_index, val_index, test_index  = random_scaffold_split_train_val_test(index = y.index.to_list(), smiles_list = self.smiles_col.to_list(), seed=0)
            elif split_type == "balanced":
                train_index, val_index, test_index  = scaffold_split_balanced_train_val_test(index = y.index.to_list(), smiles_list = self.smiles_col.to_list(), seed=0)
            elif split_type == "simple":
                train_index, val_index, test_index  = scaffold_split_train_val_test(index = y.index.to_list(), smiles_list = self.smiles_col.to_list(), seed=0)

            
            if method_chemdice.startswith("plsda"):
                print("Method name",method_chemdice)
                n_components = method_chemdice.replace("plsda_","")
                n_components = int(n_components.replace(".csv",""))
                train_dataframes,val_dataframe, test_dataframes, train_label, val_label, test_label  = save_train_test_data_s_fold(self.dataframes, self.prediction_label, train_index,val_index, test_index)
                X_train = self.fuseFeaturesTrain_plsda(n_components = n_components,  method=method_chemdice, train_dataframes = train_dataframes, train_label =train_label)
                y_train = train_label
                X_test = self.fuseFeaturesTest_plsda(n_components = n_components,  method=method_chemdice,test_dataframes = test_dataframes)
                y_test = test_label
                X_val = self.fuseFeaturesTest_plsda(n_components = n_components,  method=method_chemdice,test_dataframes = val_dataframe)
                y_val = val_label
            elif  method_chemdice.startswith("tensordecompose"):
                print("Method name",method_chemdice)
                n_components = method_chemdice.replace("tensordecompose_","")
                n_components = int(n_components.replace(".csv",""))
                train_dataframes,val_dataframe, test_dataframes, train_label, val_label, test_label  = save_train_test_data_s_fold(self.dataframes, self.prediction_label, train_index,val_index, test_index)
                X_train = self.fuseFeaturesTrain_td(n_components = n_components,  method=method_chemdice, train_dataframes = train_dataframes, train_label =train_label)
                y_train = train_label
                X_test = self.fuseFeaturesTest_td(n_components = n_components,  method=method_chemdice,test_dataframes = test_dataframes)
                y_test = test_label
                X_val = self.fuseFeaturesTest_td(n_components = n_components,  method=method_chemdice,test_dataframes = val_dataframe)
                y_val = val_label
            else:
                data = pd.read_csv(os.path.join(fused_data_path, file),index_col=0)
                X = data

                X_train, X_test, X_val = X.loc[train_index], X.loc[test_index], X.loc[val_index]
                y_train, y_test, y_val = y[train_index], y[test_index], y[val_index]
            if task_type == "classification":
                classifiers = [
                    ("Logistic Regression", LogisticRegression(), {
                        'C': [0.01, 0.1, 1, 10, 100],
                        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                        'solver': ['liblinear', 'saga']
                    }),
                    ("Decision Tree", DecisionTreeClassifier(), {
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10]
                    }),
                    ("Random Forest", RandomForestClassifier(), {
                        'n_estimators': [50, 100, 200, 300, 400],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10]
                    }),
                    ("Support Vector Machine", SVC(probability=True), {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'degree': [2, 3, 4],
                        'gamma': ['scale', 'auto']
                    }),
                    ("Naive Bayes", GaussianNB(), {
                        'var_smoothing': [1e-9, 1e-8, 1e-7]
                    }),
                    ("KNN", KNeighborsClassifier(leaf_size=1, n_neighbors=11, p=3, weights='distance'), {
                        'n_neighbors': [3, 5, 7, 9, 11],
                        'weights': ['uniform', 'distance'],
                        'leaf_size': [1, 10, 30],
                        'p': [1, 2, 3]
                    }),
                    ("NeuralNet", MLPClassifier(alpha=1, max_iter=1000), {
                        'hidden_layer_sizes': [(10,), (20,), (50,)],
                        'activation': ['relu', 'logistic', 'tanh'],
                        'solver': ['adam', 'sgd'],
                        'alpha': [0.0001, 0.001, 0.01, 1]
                    }),
                    ("QDA", QuadraticDiscriminantAnalysis(), {
                        'reg_param': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                    }),
                    ("AdaBoost", AdaBoostClassifier(), {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 1]
                    }),
                    ("Extra Trees", ExtraTreesClassifier(), {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10],
                        'max_features': ['sqrt', 'log2', None]
                    }),
                    ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.3],
                        'subsample': [0.5, 0.7, 1.0],
                        'colsample_bytree': [0.5, 0.7, 1.0]
                    })
                ]
                if models ==None:
                    models = classifiers
                else:
                    models = [clf for clf in classifiers if clf[0] in models]
            # Define models and their respective parameter grids
            elif task_type == "regression":
                regressors = [
                    ("MLP", MLPRegressor(), {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001, 0.01]}),
                    ("Kernel Ridge", KernelRidge(), {'alpha': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': [0.1, 1.0, 10.0]}),
                    ("Linear Regression", LinearRegression(), {'fit_intercept': [True, False]}),
                    ("Ridge", Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
                    ("Lasso", Lasso(), {'alpha': [0.1, 1.0, 10.0]}),
                    ("ElasticNet", ElasticNet(), {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]}),
                    ("Decision Tree", DecisionTreeRegressor(), {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}),
                    ("Random Forest", RandomForestRegressor(), {'n_estimators': [50, 100, 200, 300, 400], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}),
                    ("Gradient Boosting", GradientBoostingRegressor(), {'n_estimators': [50, 100, 200, 300, 400], 'max_depth': [3, 5, 7]}),
                    ("AdaBoost", AdaBoostRegressor(), {'n_estimators': [50, 100, 200, 300, 400], 'learning_rate': [0.01, 0.1, 1.0]}),
                    ("Support Vector Machine", SVR(), {'kernel': ['linear', 'rbf'], 'C': [0.1, 1.0, 10.0], 'epsilon': [0.1, 0.01, 0.001]}),
                    ("K Neighbors", KNeighborsRegressor(), {'n_neighbors': [3, 5, 10], 'weights': ['uniform', 'distance']}),
                    ("Gaussian Process", GaussianProcessRegressor(), {'alpha': [1e-10, 1e-9, 1e-8],'kernel': [RBF(), Matern()] })
                ]
            if models==None:
                models = regressors
            else:
                models = [regg for regg in regressors if regg[0] in models]

            parameters_dict = {model_name: [] for model_name, _, _ in models}
            best_parameters_dict = {}


            # Iterate over models
            for model_name, model, param_grid in models:
                # Iterate over hyperparameters
                for params in ParameterGrid(param_grid):
                    # Initialize and train the model with the current hyperparameters
                    model.set_params(**params)
                    model.fit(X_train, y_train)
                    
                    # Evaluate the model on the validation set
                    if  task_type == "classification":
                        y_pred_proba = model.predict_proba(X_val)[:, 1]
                        # Compute the metrics

                        auc = roc_auc_score(y_val, y_pred_proba)
                        accuracy = accuracy_score(y_val, y_pred_proba > 0.5)
                        precision = precision_score(y_val, y_pred_proba > 0.5)
                        recall = recall_score(y_val, y_pred_proba > 0.5)
                        f1 = f1_score(y_val, y_pred_proba > 0.5)
                        baccuracy = balanced_accuracy_score(y_val, y_pred_proba > 0.5)
                        mcc = matthews_corrcoef(y_val, y_pred_proba > 0.5)
                        kappa = cohen_kappa_score(y_val, y_pred_proba > 0.5)
                        # Store metrics and parameters
                        parameters_dict[model_name].append({**params,  
                                                            "auc":auc, 
                                                            'accuracy':accuracy,
                                                            "precision":precision,
                                                            "recall":recall,
                                                            "f1":f1,
                                                            "baccuracy":baccuracy,
                                                            "mcc":mcc,
                                                            "kappa":kappa })

                    elif task_type == "regression":
                        y_pred_val = model.predict(X_val)

                        mse = mean_squared_error(y_val, y_pred_val)
                        mae = mean_absolute_error(y_val, y_pred_val)
                        r2 = r2_score(y_val, y_pred_val)
                        rmse = np.sqrt(mse)

                        # Store metrics and parameters
                        parameters_dict[model_name].append({**params, 'mse': mse, 'mae': mae, 'r2': r2, 'rmse':rmse})

            # Display metrics for each model
            for model_name, params_list in parameters_dict.items():
                # print(f"Validation metrics for {model_name} with different parameteres:")
                # for params_dict in params_list:
                #     print(params_dict)
                    
                # print()
                # Get the index of the lowest MSE
                best_mse_index = min(range(len(params_list)), key=lambda i: params_list[i]['mse'])

                # Use this index to get the parameters that resulted in the lowest MSE
                
                best_parameters_dict[model_name] = params_list[best_mse_index]

                if task_type == "regression":
                    measures=  ['mse' , 'mae', 'r2','rmse']
                elif task_type == "classification":
                    measures = ["auc",
                            'accuracy',
                            "precision",
                            "recall",
                            "f1",
                            "baccuracy",
                            "mcc",
                            "kappa"]
                for measure in measures:
                    del best_parameters_dict[model_name][measure]
                    

                print("Best parameters of: " , model_name)
                print(best_parameters_dict[model_name])

            print("Testing with best parameters")
            for name, model,_ in models:
                best_parameter = best_parameters_dict[name]
                
                if method_chemdice.startswith(('pca', 'ica', 'ipca', 'cca', 'plsda')):
                    train_metrics["Model type"].append("linear")
                    test_metrics["Model type"].append("linear")
                    val_metrics["Model type"].append("linear")
                else:
                    train_metrics["Model type"].append("Non-linear")
                    test_metrics["Model type"].append("Non-linear")
                    val_metrics["Model type"].append("Non-linear")




                name = method_chemdice+"_"+name
                print("**************************   "+name + "    **************************")
                
                print(best_parameter)
                model.set_params(**best_parameter)
                model.fit(X_train, y_train)


                if task_type == "regression":

                    #print(" ######## Training data #########")
                    y_pred = model.predict(X_train)
                    # Compute the metrics
                    mse = mean_squared_error(y_train, y_pred)
                    r2 = r2_score(y_train, y_pred)
                    mae = mean_absolute_error(y_train, y_pred)
                    rmse = np.sqrt(mse)
                    # Append the metrics to the dictionary
                    train_metrics["Model"].append(name)
                    train_metrics["MSE"].append(mse)
                    train_metrics["RMSE"].append(rmse)
                    train_metrics["MAE"].append(mae)
                    train_metrics["R2 Score"].append(r2)
                
                    print(" ######## Validation data #########")
                    y_pred = model.predict(X_val)
                    # Compute the metrics
                    mse = mean_squared_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    mae = mean_absolute_error(y_val, y_pred)
                    rmse = np.sqrt(mse)
                    # Append the metrics to the dictionary
                    val_metrics["Model"].append(name)
                    val_metrics["MSE"].append(mse)
                    val_metrics["RMSE"].append(rmse)
                    val_metrics["MAE"].append(mae)
                    val_metrics["R2 Score"].append(r2)



                    #print(" ######## Test data #########")
                    y_pred = model.predict(X_test)
                    # Compute the metrics
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    # Append the metrics to the dictionary
                    test_metrics["Model"].append(name)
                    test_metrics["MSE"].append(mse)
                    test_metrics["RMSE"].append(rmse)
                    test_metrics["MAE"].append(mae)
                    test_metrics["R2 Score"].append(r2)

                
                elif task_type == "classification":
                    y_pred_proba = model.predict_proba(X_test)[:, 1] 
                    # Compute the metrics

                    auc = roc_auc_score(y_test, y_pred_proba)
                    accuracy = accuracy_score(y_test, y_pred_proba > 0.5)
                    precision = precision_score(y_test, y_pred_proba > 0.5)
                    recall = recall_score(y_test, y_pred_proba > 0.5)
                    f1 = f1_score(y_test, y_pred_proba > 0.5)
                    baccuracy = balanced_accuracy_score(y_test, y_pred_proba > 0.5)
                    mcc = matthews_corrcoef(y_test, y_pred_proba > 0.5)
                    kappa = cohen_kappa_score(y_test, y_pred_proba > 0.5)
                    # Store metrics and parameters
                    train_metrics["Model"].append(name)
                    train_metrics["AUC"].append(auc)
                    train_metrics["Accuracy"].append(accuracy)
                    train_metrics["Precision"].append(precision)
                    train_metrics["Recall"].append(recall)
                    train_metrics["f1 score"].append(f1)
                    train_metrics["Balanced accuracy"].append(baccuracy)
                    train_metrics["MCC"].append(mcc)
                    train_metrics["Kappa"].append(kappa)

                    predictions = model.predict(X_val)
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    # Compute the val_metrics
                    auc = roc_auc_score(y_val, y_pred_proba)
                    accuracy = accuracy_score(y_val, y_pred_proba > 0.5)
                    precision = precision_score(y_val, y_pred_proba > 0.5)
                    recall = recall_score(y_val, y_pred_proba > 0.5)
                    f1 = f1_score(y_val, y_pred_proba > 0.5)
                    baccuracy = balanced_accuracy_score(y_val, y_pred_proba > 0.5)
                    mcc = matthews_corrcoef(y_val, y_pred_proba > 0.5)
                    kappa = cohen_kappa_score(y_val, y_pred_proba > 0.5)
                    val_metrics["Model"].append(name)
                    val_metrics["AUC"].append(auc)
                    val_metrics["Accuracy"].append(accuracy)
                    val_metrics["Precision"].append(precision)
                    val_metrics["Recall"].append(recall)
                    val_metrics["f1 score"].append(f1)
                    val_metrics["Balanced accuracy"].append(baccuracy)
                    val_metrics["MCC"].append(mcc)
                    val_metrics["Kappa"].append(kappa)

                    #print(" ######## Validation data #########")
                    # predictions_df = pd.DataFrame({'Predictions': predictions, 'Actual': y_test})
                    # display(predictions_df)
                    # fp_df = predictions_df[(predictions_df['Predictions'] == 1) & (predictions_df['Actual'] == 0)]
                    # print("False Positive")
                    # display(fp_df)
                    # fp_prediction_list.append(fp_df.index.to_list())
                    # # False Negatives (FN): Predicted 0 but Actual 1
                    # fn_df = predictions_df[(predictions_df['Predictions'] == 0) & (predictions_df['Actual'] == 1)]
                    # print("False Negative")
                    # display(fn_df)
                    # fn_prediction_list.append(fn_df.index.to_list())
                    # predictions_dict.update({name:predictions_df})


                    #print(" ######## Test data #########")
                    predictions = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    # Compute the test_metrics
                    auc = roc_auc_score(y_test, y_pred_proba)
                    accuracy = accuracy_score(y_test, y_pred_proba > 0.5)
                    precision = precision_score(y_test, y_pred_proba > 0.5)
                    recall = recall_score(y_test, y_pred_proba > 0.5)
                    f1 = f1_score(y_test, y_pred_proba > 0.5)
                    baccuracy = balanced_accuracy_score(y_test, y_pred_proba > 0.5)
                    mcc = matthews_corrcoef(y_test, y_pred_proba > 0.5)
                    kappa = cohen_kappa_score(y_test, y_pred_proba > 0.5)
                    test_metrics["Model"].append(name)
                    test_metrics["AUC"].append(auc)
                    test_metrics["Accuracy"].append(accuracy)
                    test_metrics["Precision"].append(precision)
                    test_metrics["Recall"].append(recall)
                    test_metrics["f1 score"].append(f1)
                    test_metrics["Balanced accuracy"].append(baccuracy)
                    test_metrics["MCC"].append(mcc)
                    test_metrics["Kappa"].append(kappa)


            

            #print(test_metrics)
            # Convert dictionaries to pandas DataFrames
            test_df = pd.DataFrame(test_metrics)
            train_df = pd.DataFrame(train_metrics)
            val_df = pd.DataFrame(val_metrics)

            
            test_df.sort_values(by="R2 Score", inplace=True, ascending=False)
            train_df = train_df.loc[test_df.index]
            val_df = val_df.loc[test_df.index]

            test_df['Data'] = 'test'
            train_df['Data'] = 'train'
            val_df['Data'] = 'valid'

            Accuracy_metrics = pd.concat([test_df, train_df, val_df])


            Accuracy_metrics.to_csv(f"scaffold_split_results/Accuracy_Metrics_{method_chemdice}.csv", index=False)
            print("Done")
            print(f"scaffold_split_results/Accuracy_Metrics_{method_chemdice}.csv saved")        



    def get_accuracy_metrics(self,result_dir):
        """
        Retrieve and compile accuracy metrics from cross-validation results.

        This method aggregates the performance metrics from multiple CSV files generated by n-fold cross-validation or scafoold split.
        It calculates the mean performance metrics for each model and sorts the results based on the performance metric
        relevant to the task type (AUC for classification, R2 Score for regression).

        :return: For cross validation result : A tuple containing two DataFrames: `mean_accuracy_metrics` with the mean performance metrics for each model,
                and `Accuracy_metrics` with all the individual fold results. For scaffold split : A tuple containing three DataFrames: `test_metrics`, 
                `train_metrics` and `val_metrics`. 
        :rtype: tuple


        """
        list_of_files = os.listdir(result_dir)
        dataframes = []
        for files in list_of_files:
            if files.startswith("Accuracy_Metrics"):
                files = result_dir+"/"+files
                df = pd.read_csv(files)
                dataframes.append(df)
        
        metrics_df = pd.concat(dataframes, ignore_index=True)
        if 'Fold' not in metrics_df.columns:
            # Define the folder path
            # Combine all dataframes into a single dataframe
            
            test_df = metrics_df.loc[metrics_df['Data']=="test"]
            train_df = metrics_df.loc[metrics_df['Data']=="train"]
            val_df = metrics_df.loc[metrics_df['Data']=="valid"]

            test_df = test_df.reset_index(drop=True)
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)

            if 'AUC' in val_df.columns:
                top_models_test = pd.DataFrame()
                test_df.sort_values(by="AUC", inplace=True, ascending=False)
                train_df = train_df.loc[test_df.index]
                val_df = val_df.loc[test_df.index]
                test_df['Method'] = test_df['Model'].apply(lambda x: re.split('_|\d', x)[0])
                methods = test_df['Method'].unique()

                for method in methods:
                    # Filter rows where Model contains the method
                    method_data = test_df[test_df['Model'].str.contains(method)]
                    # Sort by R2 Score in descending order and get the top 3
                    top_method_models = method_data.nlargest(1, 'AUC')
                    # Append to the top_models DataFrame
                    top_models_test = pd.concat([top_models_test, top_method_models])

                top_models_test = top_models_test.sort_values(by='AUC', ascending=False)
                top_models_train = train_df.loc[top_models_test.index]
                top_models_val = val_df.loc[top_models_test.index]

                plot_models_barplot((top_models_train,top_models_val, top_models_test), save_dir=result_dir)
                print("plot saved to ", result_dir)
                return train_df, val_df, test_df
            
            elif 'RMSE' in val_df.columns:
                top_models_test = pd.DataFrame()
                test_df.sort_values(by="R2 Score", inplace=True, ascending=False)
                train_df = train_df.loc[test_df.index]
                val_df = val_df.loc[test_df.index]
                test_df['Method'] = test_df['Model'].apply(lambda x: re.split('_|\d', x)[0])
                methods = test_df['Method'].unique()

                for method in methods:
                    # Filter rows where Model contains the method
                    method_data = test_df[test_df['Model'].str.contains(method)]
                    # Sort by R2 Score in descending order and get the top 3
                    top_method_models = method_data.nlargest(1, 'R2 Score')
                    # Append to the top_models DataFrame
                    top_models_test = pd.concat([top_models_test, top_method_models])

                top_models_test = top_models_test.sort_values(by='R2 Score', ascending=False)
                top_models_train = train_df.loc[top_models_test.index]
                top_models_val = val_df.loc[top_models_test.index]

                plot_models_barplot((top_models_train,top_models_val, top_models_test), save_dir=result_dir)
                print("plot saved to ", result_dir)
                return train_df, val_df, test_df
        else:

            if 'AUC' in metrics_df.columns:
                top_models_test = pd.DataFrame()
                grouped_means = metrics_df.groupby('Model')["AUC"].mean()
                metrics_df = metrics_df.sort_values(by=['Model', 'Fold'], key=lambda x: x.map(grouped_means),ascending=[False, True])
                Accuracy_metrics = metrics_df

                grouped_means = metrics_df.groupby('Model').mean(numeric_only = True).reset_index()
                metrics_df = grouped_means.drop(columns=['Fold'])
                metrics_df = metrics_df.sort_values(by="AUC",ascending = False)
                mean_accuracy_metrics = metrics_df

                mean_accuracy_metrics['Method'] = mean_accuracy_metrics['Model'].apply(lambda x: re.split('_|\d', x)[0])
                methods = mean_accuracy_metrics['Method'].unique()

                for method in methods:
                    # Filter rows where Model contains the method
                    method_data = mean_accuracy_metrics[mean_accuracy_metrics['Model'].str.contains(method)]
                    # Sort by R2 Score in descending order and get the top 3
                    top_method_models = method_data.nlargest(1, 'AUC')
                    # Append to the top_models DataFrame
                    top_models_test = pd.concat([top_models_test, top_method_models])

                top_models_test = top_models_test.sort_values(by='AUC', ascending=False)
                top_Accuracy_metrics = Accuracy_metrics[ Accuracy_metrics['Model'].isin(top_models_test['Model'].to_list())] 


                plot_models_boxplot(Accuracy_metrics, save_dir=result_dir)
                print("plot saved to ", result_dir)
                return mean_accuracy_metrics, Accuracy_metrics

            elif 'RMSE' in metrics_df.columns:
                top_models_test = pd.DataFrame()
                # Group by 'Model' and calculate the mean of 'R2 Score'
                grouped_means = metrics_df.groupby('Model')["R2 Score"].mean()
                metrics_df = metrics_df.sort_values(by=['Model', 'Fold'], key=lambda x: x.map(grouped_means),ascending=[False, True])
                Accuracy_metrics = metrics_df


                grouped_means = metrics_df.groupby('Model').mean(numeric_only = True).reset_index()
                metrics_df = grouped_means.drop(columns=['Fold'])
                metrics_df = metrics_df.sort_values(by="R2 Score",ascending = False)
                mean_accuracy_metrics = metrics_df

                mean_accuracy_metrics['Method'] = mean_accuracy_metrics['Model'].apply(lambda x: re.split('_|\d', x)[0])
                methods = mean_accuracy_metrics['Method'].unique()

                for method in methods:
                    # Filter rows where Model contains the method
                    method_data = mean_accuracy_metrics[mean_accuracy_metrics['Model'].str.contains(method)]
                    # Sort by R2 Score in descending order and get the top 3
                    top_method_models = method_data.nlargest(1, 'R2 Score')
                    # Append to the top_models DataFrame
                    top_models_test = pd.concat([top_models_test, top_method_models])

                top_models_test = top_models_test.sort_values(by='R2 Score', ascending=False)
                top_Accuracy_metrics = Accuracy_metrics[ Accuracy_metrics['Model'].isin(top_models_test['Model'].to_list())] 

                plot_models_boxplot(top_Accuracy_metrics, save_dir=result_dir)
                print("plot saved to ", result_dir)
                return mean_accuracy_metrics, Accuracy_metrics       

    def fuseFeaturesTrain_plsda(self, n_components , method, train_dataframes, train_label,**kwargs):
        """
        Internal function for training of plsda fusion method.
        
        """
        prediction_label = train_label
        dataframes1 = train_dataframes
        df_list = []
        for name, df in dataframes1.items():
            df_list.append(df)
        merged_df = pd.concat(df_list, axis=1)
        
        pls_canonical = PLSRegression(n_components=n_components, **kwargs)
        #print(traindata.shape[0])
        #print(traindata.shape[1])
        #print(prediction_label.shape)
        pls_canonical.fit(merged_df, prediction_label)
        fused_df1 = pd.DataFrame(pls_canonical.transform(merged_df),
                                        columns=[f'PLS{i+1}' for i in range(pls_canonical.n_components)],
                                        index = merged_df.index) 
        self.training_pls_model = pls_canonical
        #print("Training data is fused. ")
        return fused_df1

    
    def fuseFeaturesTest_plsda(self, n_components, method, test_dataframes, **kwargs):
        """
        Internal function for testing of plsda fusion method.
        
        """
        # Iterate through the dictionary and fusion DataFrame
        # train data fusion
        dataframes1 = test_dataframes
        df_list = []
        for name, df in dataframes1.items():
            df_list.append(df)
        merged_df = pd.concat(df_list, axis=1)
        if "prediction_label" in merged_df.columns:
            # Remove the column "prediction_label" from the DataFrame
            merged_df = merged_df.drop(columns=["prediction_label"])
        #print(merged_df)
        pls_canonical = self.training_pls_model
        fused_df1 = pd.DataFrame(pls_canonical.transform(merged_df),
                                columns=[f'PLS{i+1}' for i in range(pls_canonical.n_components)],
                                index = merged_df.index) 
        fused_df1
        #print("Testing data is fused. ")
        return fused_df1

    def fuseFeaturesTrain_td(self, n_components , method, train_dataframes, train_label,**kwargs):
        """
        Internal function for training of tensordecompose fusion method.
        
        """
        df_list = []
        dataframes1 = train_dataframes
        for name, df in dataframes1.items():
            df_list.append(df)
        df_list_selected=[]
        top_features = []
        for df in df_list:
            num_features = 100
            fs = SelectKBest(score_func=f_regression, k=num_features)
            #print(df)
            X_selected = fs.fit_transform(df, train_label)
            # print(fs.get_feature_names_out())
            #print(df.columns)
            top_feature = list(fs.get_feature_names_out())
            top_features.append(top_feature)
            df_list_selected.append(df[top_feature])
        all_selected = np.array(df_list_selected)
        fused_df = apply_analysis_nonlinear4(all_selected,
                                            analysis_type=method,
                                            n_components=n_components,
                                            tol=10e-6,
                                            **kwargs)
        fused_df1 = pd.DataFrame(fused_df, index =df.index, columns = [f'TD{i+1}' for i in range(fused_df.shape[1])])
        self.top_features_train = top_features
        return fused_df1

    def fuseFeaturesTest_td(self, n_components, method, test_dataframes, **kwargs):
        """
        Internal function for testing of tensordecompose fusion method.
        
        """
        dataframes1 = test_dataframes
        top_features = self.top_features_train
        top_features = list(itertools.chain(*top_features))
        df_list = []
        for name, df in dataframes1.items():
            df_list.append(df)
        df_list_selected=[]
        for df in df_list:
            top_feature = list(set(df.columns.to_list()).intersection(set(top_features)))
            df_list_selected.append(df[top_feature])
        all_selected = np.array(df_list_selected)
        fused_df = apply_analysis_nonlinear4(all_selected,
                                            analysis_type=method,
                                            n_components=n_components,
                                            tol=10e-6,
                                            **kwargs)
        fused_df1 = pd.DataFrame(fused_df, index =df.index, columns = [f'TD{i+1}' for i in range(fused_df.shape[1])])
        #print("Testing data is fused. ")
        return fused_df1