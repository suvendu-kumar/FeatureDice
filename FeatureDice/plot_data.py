
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
from scipy import stats
import os

def plot_missing_values(df_list: List[pd.DataFrame], dataset_names: List[str], save_dir=None):
    """

    Plot stacked barcharts of missing vs non-missing values in the columns of each dataframe in a list.

    :param df_list: A list of Pandas DataFrames with missing values.
    :type df_list: List[pd.DataFrame]
    :param dataset_names: Names of the datasets corresponding to each dataframe in df_list.
    :type dataset_names: List[str]
    :param save_dir: Directory to save the plots. If None, the plots are shown. If provided, the plots are saved to this directory.
    :type save_dir: str, optional

    :output: Matplotlib figures with the barcharts.


    """
    for df, dataset_name in zip(df_list, dataset_names):
        # Get the number of missing and non-missing values in each column
        num_missing = df.isnull().sum()
        num_non_missing = df.notnull().sum()

        # Create a figure and an axis
        fig, ax = plt.subplots()

        # Plot the non-missing values as the bottom bars
        ax.bar(df.columns, num_non_missing, label="Non Missing", align='center', width=1)

        # Plot the missing values as the top bars
        ax.bar(df.columns, num_missing, bottom=num_non_missing, label="Missing")
        ax.set_xticklabels([])
        # Add some labels and a title
        ax.set_ylabel("Number of values")
        ax.set_title("Missing vs Non-Missing Values in " + dataset_name)
        ax.legend()

        # Check if running in Jupyter notebook
        if save_dir is not None:
            if not os.path.exists(save_dir): 
                # if the dir directory is not present then create it. 
                os.makedirs(save_dir) 
            plt.savefig(f'{save_dir}/{dataset_name}_missing_values_plot.png')
        else:
            plt.show()


def barplot_normality_check(df_list, dataset_names, alpha=0.05, save_dir=None):
    """

    Creates a barplot of p-values for normality test in Python for each dataframe in a list.

    :param df_list: The list of dataframes to be tested for normality.
    :type df_list: list of pandas.DataFrame
    :param dataset_names: The names of the datasets corresponding to each dataframe in df_list.
    :type dataset_names: list of str
    :param alpha: The significance level for the normality test. The default is 0.05.
    :type alpha: float, optional
    :param save_dir: The directory to save the plots. If None, the plots are shown. If provided, the plots are saved to this directory.
    :type save_dir: str, optional

    :output: Matplotlib figures with the barcharts.

    """
    # loop through each dataframe and dataset name in the list
    for df, dataset_name in zip(df_list, dataset_names):
        # initialize an empty list to store the p-values
        pvalues = []
        
        # loop through each column of the dataframe
        for col in df.columns:
            # perform the D'Agostino's K^2^ test on the column
            k2, p = stats.normaltest(df[col])
            # append the p-value to the list
            pvalues.append(p)
        
        # create a figure and an axis
        fig, ax = plt.subplots()
        
        # create a barplot of the p-values
        ax.bar(df.columns, pvalues, color='blue')
        # add a horizontal line at the significance level
        ax.axhline(y=alpha, color='red', linestyle='--')
        # add labels and a title
        ax.set_xlabel('Columns')
        ax.set_ylabel('P-values')
        ax.set_title('Barplot of P-values for Normality Test in ' + dataset_name)
        
        # check if a save directory is provided
        if save_dir is not None:
            # create the directory if it does not exist
            if not os.path.exists(save_dir): 
                os.makedirs(save_dir) 
            # save the plot to the directory
            plt.savefig(f'{save_dir}/{dataset_name}_normality_plot.png')
        else:
            # show the plot
            plt.show()




def plot_model_boxplot(metrics_df, save_dir = None):
    """
    Plots the performance metrics of different models on a boxplot chart.

    :param metrics_df: The dataframe containing the performance metrics of different models.
    :type metrics_df: pd.DataFrame
    :param save_dir: The directory where the plot will be saved. If None, the plot will be displayed without being saved.
    :type save_dir: str, optional

    :output: Matplotlib figures with the boxplot charts.

    .. note::  
        The metrics are : AUC, Accuracy, Precision, Recall, f1 score, Balanced accuracy, MCC and Kappa for classification. R2 Score and MSE for regression.
        
        - **AUC**: The area under the receiver operating characteristic curve, which measures the trade-off between the true positive rate and the false positive rate.
        - **Accuracy**: The proportion of correctly classified instances among the total number of instances.
        - **Precision**: The proportion of correctly classified positive instances among the predicted positive instances.
        - **Recall**: The proportion of correctly classified positive instances among the actual positive instances.
        - **f1 score**: The harmonic mean of precision and recall, which balances both metrics.
        - **Balanced accuracy**: The average of recall obtained on each class, which is useful for imbalanced datasets.
        - **MCC**: The Matthews correlation coefficient, which measures the quality of a binary classification, taking into account true and false positives and negatives.
        - **Kappa**: The Cohen's kappa coefficient, which measures the agreement between two raters, adjusting for the chance agreement.
        - **R2 Score**: The statistical measure that indicates the proportion of variance in the dependent variable that's predictable from the independent variable(s). 
        - **MSE(Mean Squared Error)**: The average of the squared differences between the actual and predicted values, used as a loss function in regression problems.
    """


    for metric in ['AUC', 'Accuracy', 'Precision', 'Recall','f1 score', 'Balanced accuracy', 'MCC', 'Kappa']:
        plt.figure(figsize=(10, 6))

        # Creating the box plot
        sns.boxplot(x='Model', y=metric, data=metrics_df, palette="Set3")

        # Adding jitter for the Fold information with stripplot
        # Note: 'dodge=True' allows the jitter to be properly aligned with each box
        sns.stripplot(x='Model', y=metric, data=metrics_df, hue='Fold', dodge=True, jitter=True, marker='o', alpha=0.8, edgecolor='gray')

        # Enhancing the plot
        plt.title(f'Box Plot with {metric} values for different methods at different folds')
        plt.legend(title='Fold', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        # Save the figure
        # check if a save directory is provided
        if save_dir is not None:
            # create the directory if it does not exist
            if not os.path.exists(save_dir): 
                os.makedirs(save_dir) 
            # save the plot to the directory
            plt.savefig(f"{save_dir}/metrics{metric}.png")
        else:
            # show the plot
            plt.show()

from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

def plot_model_metrics(metrics_df, save_dir=None):
    """
    Plot the performance metrics of different models on a bar chart.

    :param metrics_df: A pandas DataFrame with the model names, types, and metrics as columns.
    :type metrics_df: pd.DataFrame
    :param save_dir: The directory to save the plots. If None, the plots are shown instead.
    :type save_dir: str or None

    .. note::  
        The metrics are : AUC, Accuracy, Precision, Recall, f1 score, Balanced accuracy, MCC and Kappa for classification. R2 Score and MSE for regression.
        
        - **AUC**: The area under the receiver operating characteristic curve, which measures the trade-off between the true positive rate and the false positive rate.
        - **Accuracy**: The proportion of correctly classified instances among the total number of instances.
        - **Precision**: The proportion of correctly classified positive instances among the predicted positive instances.
        - **Recall**: The proportion of correctly classified positive instances among the actual positive instances.
        - **f1 score**: The harmonic mean of precision and recall, which balances both metrics.
        - **Balanced accuracy**: The average of recall obtained on each class, which is useful for imbalanced datasets.
        - **MCC**: The Matthews correlation coefficient, which measures the quality of a binary classification, taking into account true and false positives and negatives.
        - **Kappa**: The Cohen's kappa coefficient, which measures the agreement between two raters, adjusting for the chance agreement.
        - **R2 Score**: The statistical measure that indicates the proportion of variance in the dependent variable that's predictable from the independent variable(s). 
        - **MSE(Mean Squared Error)**: The average of the squared differences between the actual and predicted values, used as a loss function in regression problems.
    """

    if metrics_df.shape[1] > 8:
        matric_list = ['AUC', 'Accuracy', 'Precision', 'Recall','f1 score', 'Balanced accuracy', 'MCC', 'Kappa']
    else:
        matric_list = ["R2 Score" , "MSE"]
    for metric in matric_list:
        fig, ax = plt.subplots()
        # Get unique values from the 'Model type' column
        unique_categories = metrics_df['Model type'].unique()
        # Create a dictionary mapping each unique category to a unique color
        color_mapping = {category: plt.cm.viridis(i / len(unique_categories)) for i, category in enumerate(unique_categories)}
        # Map the colors to the original 'Model type' column and convert to RGBA format
        metrics_df['Color'] = metrics_df['Model type'].map(color_mapping).apply(to_rgba)
        # Create a bar plot
        bars = ax.bar(metrics_df['Model'], metrics_df[metric], color=metrics_df['Color'], label=metrics_df['Model type'])
        # Add labels and title
        ax.set_xlabel('Models')
        ax.set_ylabel(f'{metric} Values')
        ax.set_title(f'Bar Plot with {metric} values for different methods')
        # Rotate x-axis labels by 45 degrees
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
        # Add legend
        #ax.legend()
        # Adjust layout
        fig.tight_layout()
        # Save the figure
        if save_dir is not None:
            # create the directory if it does not exist
            if not os.path.exists(save_dir): 
                os.makedirs(save_dir) 
            # save the plot to the directory
            fig.savefig(f"{save_dir}/metrics{metric}.png")
        else:
            # show the plot
            plt.show()

    legend_handles = [Line2D([0], [0], color=color, lw=4, label=category) for category, color in color_mapping.items()]

    # Create legend without a plot
    legend_fig, legend_ax = plt.subplots(figsize=(6, 1))
    legend_ax.set_axis_off()
    legend_ax.legend(handles=legend_handles, loc='center', ncol=len(unique_categories))

    # Save the legend as a separate image
    if save_dir == None:
        legend_fig.show()
    else:
        legend_fig.savefig(f"{save_dir}/legend.png", bbox_inches='tight')




