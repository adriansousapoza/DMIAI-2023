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
from sklearn.preprocessing import QuantileTransformer, StandardScaler
#from category_encoders import LeaveOneOutEncoder, TargetEncoder

import optuna
from optuna.samplers import TPESampler
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import MedianPruner
import lightgbm as lgb
import xgboost as xgb

from utils import *


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
    else:
        train_features = df.drop(['label'], axis=1).astype('float')
        
    train_labels = df['label']
    X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def evaluate_classification_results(estimator, X_train, X_val, y_train, y_val, X_test = None, y_test = None, method_name = '', plot = False, booster=False):
         
        N = 2 if X_test is None else 3

        X = [X_train, X_val] if X_test is None else [X_train, X_val, X_test]
        Y = [y_train, y_val] if y_test is None else [y_train, y_val, y_test]

        fpr_list, tpr_list = [], []
        auc_score = []
        log_loss_score = []
        acc_score = []

        for x,y in zip(X,Y):
            y = y.values.reshape(-1)

            if booster:
                pred = estimator.predict(x)
            else:
                pred = estimator.predict_proba(x)[:,1]
            
            fpr, tpr, _ = roc_curve(y, pred)
            fpr_list.append(fpr), tpr_list.append(tpr)
            auc_score.append(auc(fpr, tpr))
            log_loss_score.append(log_loss(y, pred))
            acc_score.append(accuracy_score(y, pred > 0.5))

        print("binary/acc/AUC train: ", log_loss_score[0], acc_score[0], auc_score[0])
        print("binary/acc/AUC val: ", log_loss_score[1], acc_score[1], auc_score[1])

        if N == 3:
            print("binary/acc/AUC test: ", log_loss_score[2], acc_score[2], auc_score[2])

        if plot:
            fig, ax = plt.subplots()
            ax.plot(fpr_list[0], tpr_list[0], label=f'Train (AUC = {auc_score[0]:5.3f})')
            ax.plot(fpr_list[1], tpr_list[1], label = f'Val (AUC = {auc_score[1]:5.3f})')
            if N == 3:
                ax.plot(fpr_list[2], tpr_list[2], label = f'Test (AUC = {auc_score[2]:5.3f})')
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
    with open('models/best_params_lgb_clf.pkl', 'wb') as fp:
        pickle.dump(lgb_clf_params, fp)

    ## save beset model
    lgb_clf.booster_.save_model('models/lgb_classifier_best_model.json')
    return

### MAIN ---------------------------------------------------------------------------------------


def main():

    feature_csv_path = 'data_processed/data_all_balanced_many_features.csv'  #'data/final_training_data_features.csv' # None #'data_processed\\data_all_features.csv'
    string_csv_path = 'data/final_training_data.csv'  #'data_processed\\data_all.csv'
    features_validation_path = 'data/final_validation_data_features.csv'
    validation_path = 'valz_labeled.csv'

    load_model = False
    hyperoptimization = False
    include_val = True

    # Set sklearn-style scaler/transformer if any
    scaler = None #QuantileTransformer(output_distribution='normal') #StandardScaler()   #None #QuantileTransformer(output_distribution='normal')

    if feature_csv_path is not None:
        df = pd.read_csv(feature_csv_path, header=0)
        df = df.drop(['text', 'Unnamed: 0'], axis=1)
    else:
        df = pd.read_csv(string_csv_path, header = 0, names=['idx','text','label'])
        df.drop(['idx'], axis=1, inplace=True)
        df = preprocess(df)

    if features_validation_path is not None:
        val_features = pd.read_csv(features_validation_path, header=0)
        val_labels = val_features['label']
        val_features = val_features.drop(['text', 'Unnamed: 0'], axis=1)
    else:
        df_val = pd.read_csv(validation_path, header=0, names=['idx','text','label'])
        df_val.drop(['idx'], axis=1, inplace=True)
        val_features = preprocess(df_val)
        val_labels = val_features['label']

    test_final = val_features.sample(frac=0.2, random_state=42)
    test_labels = test_final['label']
    test_final = test_final.drop(['label'], axis=1)
    val_features = val_features.drop(test_final.index)

    df = pd.concat([df, val_features], ignore_index=True)
    df = df.sample(frac=1, random_state=42)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = scale_and_split_data(df, scaler = scaler, test_size = 0.1, val_size = 0.15, random_state = 42)

    lgb_kwargs = dict(boosting_type='gbdt', num_leaves=120, \
                                max_depth=4, learning_rate=0.2, n_estimators=100, \
                                objective='binary', min_split_gain=0.0,\
                                min_child_samples=5, subsample = 1.0,
                                reg_alpha=0.0, reg_lambda=0.0, \
                                n_jobs=-1, importance_type = 'split') 

    if load_model:
        lgb_clf = lgb.Booster(model_file='models/lgb_classifier.json')
        booster = True
    else:
        lgb_clf = lgb.LGBMClassifier(**lgb_kwargs)
        lgb_clf.fit(X_train, y_train, eval_set = [(X_val, y_val)])
        booster = False
        lgb_clf.booster_.save_model('models/lgb_classifier.json')

    if scaler is not None:
        scaled_test_features_arr = scaler.transform(test_final.values)
        test_final = pd.DataFrame(scaled_test_features_arr, index = test_final.index, \
                                            columns = test_final.columns)

    #val_labels = pd.read_csv('data_processed\\validation_labeled.csv')['labels'].values
    val_preds = lgb_clf.predict(test_final)
    accu =np.isclose(val_preds, test_labels).sum()/len(val_preds)

    print("ON VAL ..... ",accu)
    print(accuracy_score(test_labels.values, val_preds))

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
