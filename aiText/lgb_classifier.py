# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import pickle
import os, sys
import numpy as np
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler


from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import roc_curve, auc, log_loss, accuracy_score
from sklearn.preprocessing import QuantileTransformer
#from category_encoders import LeaveOneOutEncoder, TargetEncoder

import optuna
from optuna.samplers import TPESampler
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import MedianPruner
import lightgbm as lgb

from utils import *

nltk.download('punkt');


## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


pd.options.mode.chained_assignment = None

## Set plotting style and print options
sns.set_theme()
sns.set_style("darkgrid")
sns.set_context("paper") #Possible are paper, notebook, talk and poster

d = {'lines.linewidth': 2, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 12, 'ytick.labelsize': 12,\
     'legend.fontsize': 15, 'font.family': 'serif', 'figure.figsize': (9,6)}
d_colors = {'axes.prop_cycle': cycler(color = ['teal', 'navy', 'coral', 'plum', 'purple', 'olivedrab',\
         'black', 'red', 'cyan', 'yellow', 'khaki','lightblue'])}
rcParams.update(d)
rcParams.update(d_colors)
np.set_printoptions(precision = 5, suppress=1e-10)


### FUNCTIONS ----------------------------------------------------------------------------------

def scale_and_split_data(df, scaler = None, test_size = 0.15, val_size = 0.15, random_state = 42):
    """
    Scales and splits data into training, validation and test data
    """
    if scaler is not None:
        scaled_train_features_arr = scaler.fit_transform(df.drop(['label'], axis=1).values)
        train_features = pd.DataFrame(scaled_train_features_arr, index = df.index, \
                                            columns = df.drop(['label'], axis=1).columns)
        
    train_labels = df['label']
    X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def evaluate_classification_results(estimator, X_train, X_val, y_train, y_val, X_test = None, y_test = None, method_name = '', plot = False, booster=False):
         
        N = 2 if X_test is None else 3

        X = [X_train, X_val] if X_test is None else [X_train, X_val, X_test]
        y = [y_train, y_val] if y_test is None else [y_train, y_val, y_test]

        pred = np.zeros((N, len(y_val)))
        fpr, tpr = np.zeros(N) = np.zeros(N)
        auc_score = np.zeros(N)
        log_loss_score = np.zeros(N)
        acc_score = np.zeros(N)

        for i in range(N):
            if booster:
                pred[i] = estimator.predict(X[i])
            else:
                pred[i] = estimator.predict_proba(X[i])[:,1]

            fpr[i], tpr[i], _ = roc_curve(y[i], pred[i])                
            auc_score[i] = auc(fpr[i],tpr[i])
            log_loss_score[i] = log_loss(y[i], pred[i])
            acc_score[i] = accuracy_score(y[i], pred[i] > 0.5) if booster else estimator.score(y[i], pred[i] > 0.5)
 
            print("binary/acc/AUC train: ", benchmark_stats_train)
            print("binary/acc/AUC val: ", benchmark_stats_val)
            if N == 3:
                print("binary/acc/AUC test: ", benchmark_stats_test)
            
            if plot:
                fig, ax = plt.subplots()
                ax.plot(fpr_train, tpr_train, label=f'Train (AUC = {auc_score_train:5.3f})')
                ax.plot(fpr_val, tpr_val, label = f'Val (AUC = {auc_score_val:5.3f})')
                if N == 3:
                    ax.plot(fpr_test, tpr_test, label = f'Test (AUC = {auc_score_test:5.3f})')
                ax.set(title=f'{method_name} ROC', xlabel = 'FPR', ylabel = 'TPR')
                ax.legend() 
            return
    
def objective(trial, method, param_wrapper, scoring, X_train, y_train):

    params = param_wrapper(trial)
    clf = method(**params)

    cv_results = cross_validate(clf, X_train, y_train, scoring=scoring, cv=5) #, fit_params=xgb_params)
    print(cv_results['test_score'])
    return cv_results['test_score'].mean()
            
def hyperoptizing_lgbm_clf(X_train, X_val, y_train, y_val, parameter_wrapper, n_trials = 25, verbose = True):
           
    # create optimization object
    study = optuna.create_study(direction="maximize",sampler=TPESampler(),pruner=MedianPruner(n_warmup_steps=50))

    # create objective function to pass and optimize
    restricted_objective = lambda trial: objective(trial, lgb.LGBMClassifier, parameter_wrapper, \
                        scoring='neg_log_loss', X_train = X_train, y_train = y_train)
    study.optimize(restricted_objective, n_trials= n_trials, show_progress_bar=False)

    
    lgb_clf_params = {}
    lgb_clf_params.update(study.best_trial.params)

    if verbose:
        print("Optimized parameters", lgb_clf_params)
        print(study.best_trial.params)
        print(study.best_trial.values)


    ## Fit using the best parameters and evaluate results
    lgb_clf = lgb.LGBMClassifier(**lgb_clf_params)
    lgb_clf.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose=0)

    evaluate_classification_results(lgb_clf, X_train, \
                                    X_val, y_train, y_val, method_name='LGB', plot=True)

    # Save best parameters
    with open('best_params_lgb_clf.pkl', 'wb') as fp:
        pickle.dump(lgb_clf_params, fp)

    ## save beset model
    lgb_clf.booster_.save_model('lgb_classifier.json')


### MAIN ---------------------------------------------------------------------------------------


def main():

    feature_csv_path =  'data_features.csv'
    string_csv_path = 'data.csv'
    load_model = False
    hyperoptimization = False

    # Set sklearn-style scaler/transformer if any
    scaler = None

    if feature_csv_path is not None:
        df = pd.read_csv(feature_csv_path)
    else:
        df = pd.read_csv(string_csv_path)
        df = preprocess(df)

    
    X_train, X_val, X_test, y_train, y_val, y_test, _ = scale_and_split_data(df, scaler = scaler, test_size = 0.15, val_size = 0.15, random_state = 42)

    lgb_kwargs = dict(boosting_type='gbdt', num_leaves=45, \
                                max_depth=3, learning_rate=0.2, n_estimators=175, \
                                objective='binary', min_split_gain=0.0,\
                                min_child_samples=1, subsample = 0.98,
                                reg_alpha=0.05, reg_lambda=0.05, \
                                n_jobs=-1, importance_type = 'split') 

    if load_model:
        lgb_clf = lgb.Booster(model_file='lgb_classifier.json')
        booster = True

    else:
        lgb_clf = lgb.LGBMClassifier(**lgb_kwargs)
        lgb_clf.fit(X_train, y_train, eval_set = [(X_val, y_val)])
        booster = False
        lgb_clf.booster_.save_model('lgb_classifier.json')

    print(lgb_clf.score(X_test, y_test))
    evaluate_classification_results(lgb_clf, X_train, \
                                                X_val, y_train, y_val, X_test, y_test, method_name='LGB', plot=True, booster=booster)

    if hyperoptimization:

        # define wrapper holding the model parameters
        def lgb_clf_wrapper(trial):

            importance_types = ["gain", "split"]
            importance_type = trial.suggest_categorical("importance_type", importance_types)

            # set constant parameters
            lgb_clf_kwargs = dict(boosting_type='gbdt', min_split_gain=0.0,\
                                reg_alpha=0.0, reg_lambda=0.0,
                                objective='binary', n_jobs=-1)
            # set parameters to vary                  
            lgb_clf_kwargs_var =  {#'subsample': trial.suggest_float('subsample', 0.6,1),
                        'n_estimators': trial.suggest_int('n_estimators',250,400),
                        'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.4),
                        'max_depth': trial.suggest_int('max_depth', 4,10), \
                        'num_leaves': trial.suggest_int('num_leaves',50, 200), \
                        'subsample': trial.suggest_float('subsample', 0.75,1),\
                        'importance_type': importance_type,}
            
            lgb_clf_kwargs.update(lgb_clf_kwargs_var)
            return lgb_clf_kwargs

        hyperoptizing_lgbm_clf(X_train, X_val, y_train, y_val, \
                                    lgb_clf_wrapper, scoring='neg_log_loss', n_trials = 25, verbose = True)



if __name__ == '__main__':
    main()
