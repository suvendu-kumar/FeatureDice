
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
from scipy import stats
import os





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




