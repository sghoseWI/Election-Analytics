import pandas as pd
import numpy as np

from itertools import product, chain
import sklearn
from sklearn import preprocessing, svm, metrics, tree, decomposition, svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split, ParameterGrid, train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, normalize
import random
from scipy import optimize
from sklearn.metrics import recall_score, f1_score, precision_score, roc_auc_score, accuracy_score
from sklearn.metrics import make_scorer

def run_simple_loop(x_train, x_test, y_train, y_test, grid_size = 'test', models_to_run=['LR']):

    # define grid to use: test, small, large
    clfs, grid = define_clfs_params(grid_size)

    # define models to run
    models_to_run=models_to_run

    # call clf_loop and store results in results_df
    results_df = clf_loop(models_to_run, clfs, grid, x_train, x_test, y_train, y_test)

    return results_df

def filter_df_by_date_range(df, date_var, start, end):
    df = df[(df[date_var] >= start) & (df[date_var] < end)]
    return df

def split_labels(df, outcome_var, exclude = [], keep_cols = False):
    if not keep_cols:
        skips = [outcome_var] + exclude
        Xs = df.drop(skips, axis = 1)
    else:
        Xs = df[keep_cols]

    Ys = df[outcome_var]

    return Xs, Ys

def find_best_model(results_df, criteria='p_at_1'):
    best_val = 0
    best_model = []
    best_params = []
    for idx, m in results_df.iterrows():
        if m[criteria] > best_val:
            best_val = m[criteria]
            best_model = m['clf']
            best_params = m['parameters']
            
    return best_model

def LR():
    return LinearRegression()

def define_clfs_params(grid_size):
    """Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    """

    clfs = {
        'LR': LinearRegression()}

    small_grid = {
    
    'LR': { 'fit_intercept': [True], 'normalize': [False], 'copy_X': [True], 'n_jobs':[None]},
           }
    
    return clfs, small_grid

# a set of helper function to do machine learning evalaution

def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)

    precision = precision_score(y_true, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)

    recall = recall_score(y_true_sorted, preds_at_k)
    return recall

def scores_at_k(y_true, y_scores, k):
    '''
    Calculate precision, recall, and f1 score at a given threshold
    '''
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    recall = recall_score(y_true, preds_at_k)
    f1 = f1_score(y_true, preds_at_k)
    
    return precision, recall, f1

def clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test, thresholds = [5, 10, 20]):
    """
    Runs the loop using models_to_run, clfs, gridm and the data
    """

    result_cols = ['model_type','clf', 'parameters', 'baseline_p', 'auc-roc']

    # define columns for metrics at each threshold specified in function call
    result_cols += list(chain.from_iterable(('p_at_{}'.format(threshold), 
                        'r_at_{}'.format(threshold), 
                        'f1_at_{}'.format(threshold)) for threshold in thresholds))

    results_df =  pd.DataFrame(columns=result_cols)

    TREE_COUNTER = 0

    for n in range(1, 2):
        # create training and valdation sets
        for index, clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            model_count = 0
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                
                    results_list = [models_to_run[index], clf, p, 
                    precision_at_k(y_test_sorted, y_pred_probs_sorted, 100.0),
                    roc_auc_score(y_test, y_pred_probs)]

                    for threshold in thresholds:
                        precision, recall, f1 = scores_at_k(y_test_sorted, 
                            y_pred_probs_sorted, threshold)
                        results_list += [precision, recall, f1]

                    results_df.loc[len(results_df)] = results_list
                except IndexError as e:
                    print('Error:',e)
                    continue
    return results_df
