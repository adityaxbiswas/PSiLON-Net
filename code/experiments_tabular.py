

import os
import shutil
from os import path
def clear_folder(folder):
    for root, dirs, files in os.walk(folder):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
def create_folder(folder):
    if not path.exists(folder):
        os.makedirs(folder)

project_folder = path.abspath(path.join(path.sep,"Projects", "PSiLON-Net"))
# clear log and checkpoint folders. create if doesnt exist
lightning_logs_folder = path.join(project_folder, 'code', 'lightning_logs')
checkpoint_folder = path.join(project_folder, "code", "model_checkpoints")
create_folder(lightning_logs_folder)
create_folder(checkpoint_folder)
clear_folder(lightning_logs_folder)
clear_folder(checkpoint_folder)


# setup openml
import openml
with open(path.join(project_folder, "openml_apikey.txt")) as apikey_file:
    apikey = apikey_file.read()
openml.config.apikey = apikey
openml.config.cache_directory = path.join(project_folder, "datasets")

import math
import itertools
import pandas as pd
import numpy as np
from scipy.stats import iqr
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import log_loss


from torch import nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning as L
import logging
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

# import custom functions/classes
from NN_utils import *
from NeuralNetwork import *
###################################################################################
### Main neural network training function
def get_NN_result(model, compute_loss, compute_eval,
                   dl_train, dl_val, dl_test,
                   lambda_list, verbose=True, 
                   use_es=False, compile=True):
    
    recorder = HyperparameterRecorder({'lambda_': lambda_list}, 
                                    verbose = verbose)
    mc_dict = {}
    while not recorder.is_complete():
        # create lightning model
        hyperparams = recorder.next()
        lambda_ = hyperparams['lambda_']
        if compile:
            model = torch.jit.script(model)
        litmodel = LitNetwork(model, lambda_=lambda_,
                            compute_loss=compute_loss,
                            compute_eval=compute_eval,
                            n_batches = n_batches,
                            n_epochs = epochs,
                            lr = 2e-3)
        
        # define trainer
        mc_dir = path.join(checkpoint_folder, f"model-reg-{lambda_}")
        if use_es:
            es = EarlyStopping(monitor='val_loss', mode='min', 
                            patience=50, verbose=False, min_delta=0.0001)
            mc_dict[lambda_] = ModelCheckpoint(dirpath = mc_dir, save_top_k=1, 
                                            monitor='val_loss', mode="min", 
                                            every_n_epochs=1, save_on_train_epoch_end=True)
            check_val_every_n_epoch = 1
            callbacks = [mc_dict[lambda_], es]
        else:
            check_val_every_n_epoch = 20
            callbacks = []
        trainer = L.Trainer(callbacks=callbacks, 
                            max_epochs = epochs,
                            accelerator=accelerator,
                            num_sanity_val_steps=0,
                            log_every_n_steps=n_batches,
                            check_val_every_n_epoch = check_val_every_n_epoch,
                            enable_checkpointing=True,
                            enable_progress_bar=False,
                            enable_model_summary=False)
        
        # train, predict on val, and record
        if use_es:
            trainer.fit(litmodel, dl_train, dl_val)
            performance = mc_dict[lambda_].best_model_score.item()
        else:
            trainer.fit(litmodel, dl_train, dl_val)
            trainer.save_checkpoint(path.join(mc_dir, 'model_final.ckpt'))
            performance = trainer.test(litmodel, dl_val, verbose = False)[0]['test_loss']
            if verbose:
                test_performance = trainer.test(litmodel, dl_test, verbose = False)[0]['test_loss']
                print(f"Test Loss: {np.round(test_performance, 5)}")
        recorder.record(performance)

    # load best model and predict on test set
    hyperparams = recorder.get_best()
    lambda_ = hyperparams['lambda_']
    print(f"Optimal value of Lambda: {lambda_}")
    if use_es:
        final_model_path = mc_dict[lambda_].best_model_path
    else:
        mc_dir = path.join(checkpoint_folder, f"model-reg-{lambda_}")
        final_model_path = path.join(mc_dir, 'model_final.ckpt')
    litmodel = LitNetwork.load_from_checkpoint(final_model_path,
                                            model = model,
                                            lambda_=lambda_,
                                            compute_loss=compute_loss,
                                            compute_eval=compute_eval,
                                            n_batches = n_batches,
                                            n_epochs = epochs)
    performance = trainer.test(litmodel, dl_test, verbose = False)[0]['test_loss']
    nsparsity = litmodel.get_sparsity(near=True)

    # clear checkpoint and logs folders
    clear_folder(lightning_logs_folder)
    clear_folder(checkpoint_folder)
    return performance, nsparsity
###################################################################################
# smaller helper function/tools

RMSE = lambda Y,Y_hat: np.sqrt(np.mean((Y-Y_hat)**2))
def select_continuous_features(feature_names, categorical_indicator):
    features_categorical = itertools.compress(feature_names, categorical_indicator)
    return list(set(feature_names) - set(features_categorical))

repeat_dataset_names = ['delays_zurich_transport', 'diamonds', 'Brazilian_houses',
                        'Bike_Sharing_Demand', 'nyc-taxi-green-dec-2016',
                        'house_sales', 'medical_charges', 'abalone', 'electricity',
                        'eye_movements', 'covertype', 'default-of-credit-card-clients']

######################################################################################
### START SCRIPT
n_keep = 20000
train_size = 2000

rf_performances = {}
lr_performances = {}
psilon_performances = {}
standard_performances = {}
psilon_nsparsities = {}
standard_nsparsities = {}


# Choose which of the 4 suites you want to run
#SUITE_ID = 335 # Regression on numerical and categorical features
SUITE_ID = 336 # Regression on numerical features
#SUITE_ID = 334 # Classification on numerical and categorical features
#SUITE_ID = 337 # Classification on numerical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite
classification = (SUITE_ID == 334) or (SUITE_ID == 337)

# Run the suite
for task_id in benchmark_suite.tasks:  # iterate over all tasks
    # download/load the dataset
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()

    # exclude certain datasets
    if (SUITE_ID == 336) or (SUITE_ID == 337):
        if dataset.name in repeat_dataset_names:
            continue
    if classification:
        if dataset.name in ['pol', 'house_16H']: # already regression datasets
            continue
    
    # load data.  keep only if 10 or more predictors
    X, y, categorical_indicator, feature_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    if len(X.columns) < 10:
        continue
    if y.dtype.name == "category":
        y = y.cat.codes.copy()
    print("Loaded dataset: " + dataset.name)
    
    # randomly reduce data in size and train/test split
    n = min(len(y),n_keep)
    shuffler = np.random.permutation(len(y))
    X, y = X.iloc[shuffler,:], y.iloc[shuffler]
    X = X.iloc[:n,:]
    y = y.iloc[:n]
    X_train, X_test = X.iloc[:train_size,:], X.iloc[train_size:,:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]


    # preprocess the data and split into train/val
    features_continuous = select_continuous_features(feature_names, categorical_indicator)
    transformer = ColumnTransformer([('scaler', StandardScaler(), features_continuous)],
                                    remainder = 'passthrough')
    X_train = transformer.fit_transform(X_train).astype(np.float32)
    X_test = transformer.transform(X_test).astype(np.float32)
    if not classification:
        y_median, y_qd = y.median(), iqr(y)/2
        y_train = (y_train-y_median)/y_qd
        y_test = (y_test-y_median)/y_qd
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                      test_size=0.5,
                                                      shuffle=False)
    

    ######################################################################################
    ### Baseline model training

    
    # Ridge Regression
    recorder = HyperparameterRecorder({'alpha': [1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,
                                                 1e-3,3e-3,1e-2,3e-2,1e-1,0.3,1,3,10,30,100]},
                                        verbose = False)
    while not recorder.is_complete():
        hyperparams = recorder.next()
        if classification:
            model = LogisticRegression(C=1/hyperparams['alpha'])
            model.fit(X_train, y_train)
            y_hat_val = model.predict_proba(X_val)[:,1]
            recorder.record(log_loss(y_val, y_hat_val))
        else:
            model = Ridge(alpha=hyperparams['alpha'])
            model.fit(X_train, y_train)
            y_hat_val = model.predict(X_val)
            recorder.record(RMSE(y_val, y_hat_val))

    hyperparams = recorder.get_best()
    if classification:
        model = LogisticRegression(C=1/hyperparams['alpha'])
        model.fit(X_train, y_train)
        y_hat_test = model.predict_proba(X_test)[:,1]
        performance = log_loss(y_test, y_hat_test)
    else:
        model = Ridge(alpha = hyperparams['alpha'])
        model.fit(X_train, y_train)
        y_hat_test = model.predict(X_test)
        performance = RMSE(y_test, y_hat_test)

    lr_performances[dataset.name] = performance
    print(f"Finished Ridge Regression Experiment: {np.round(performance, 5)}")
    
    
    # Random Forest
    recorder = HyperparameterRecorder({'min_samples_leaf': [4,8,12,16,20,30,
                                                            40,50,75,100,150,200]},
                                      verbose = False)
    while not recorder.is_complete():
        hyperparams = recorder.next()
        if classification:
            model = RFC(n_estimators = 1000, 
                        min_samples_leaf = hyperparams['min_samples_leaf'],
                        n_jobs = -1)
            model.fit(X_train, y_train)
            y_hat_val = model.predict_proba(X_val)[:,1]
            recorder.record(log_loss(y_val, y_hat_val))
        else:
            model = RFR(n_estimators = 1000, 
                        min_samples_leaf = hyperparams['min_samples_leaf'],
                        n_jobs = -1)
            model.fit(X_train, y_train)
            y_hat_val = model.predict(X_val)
            recorder.record(RMSE(y_val, y_hat_val))

    hyperparams = recorder.get_best()
    if classification:
        model = RFC(n_estimators = 1000, 
                    min_samples_leaf = hyperparams['min_samples_leaf'],
                    n_jobs = -1)
        model.fit(X_train, y_train)
        y_hat_test = model.predict_proba(X_test)[:,1]
        performance = np.mean(log_loss(y_test, y_hat_test))
    else:
        model = RFR(n_estimators = 1000, 
                    min_samples_leaf = hyperparams['min_samples_leaf'],
                    n_jobs = -1)
        model.fit(X_train, y_train)
        y_hat_test = model.predict(X_test)
        performance = RMSE(y_test, y_hat_test)

    rf_performances[dataset.name] = performance
    print(f"Finished Random Forest Experiment: {np.round(performance, 5)}")
    


    ################################################################################
    # Neural Network training
    
    if classification:
        compute_loss = nn.BCEWithLogitsLoss()
        compute_eval = nn.BCEWithLogitsLoss()
    else:
        compute_loss = nn.MSELoss()
        compute_eval = RMSELoss()
    epochs = 1000
    use_es = False # es = early stopping
    hidden_size = 500
    n_hidden = 3
    verbose = True
    accelerator = 'gpu'
    compile = True
    n_batches = 5
    batch_size = int(math.ceil(len(X_train)/n_batches))

    dl_train = convert_to_dataloader(X_train, y_train.values,
                                     batch_size=batch_size,
                                     shuffle = True)
    dl_val = convert_to_dataloader(X_val, y_val.values)
    dl_test = convert_to_dataloader(X_test, y_test.values)

    lambda_list = [5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 
                   1e-2, 2.5e-2, 5e-2, 1e-1, 2.5e-1, 5e-1]


    #PSiLONNet
    model = NormNet(X_train.shape[1], hidden_size, 1, n_hidden = n_hidden,
                    share=True, use_bias=True, use_l2_wn=False, use_1pathnorm=True)
    performance, nsparsity = get_NN_result(model, compute_loss, compute_eval,
                                            dl_train, dl_val, dl_test,
                                            lambda_list,
                                            verbose=verbose, 
                                            use_es=use_es,
                                            compile=compile)
    psilon_performances[dataset.name] = performance
    psilon_nsparsities[dataset.name] = nsparsity
    print(f"\nFinished PSiLON Net Experiment: Performance: {np.round(performance, 5)}")
    print(f"Finished PSiLON Net Experiment: NS: {np.round(nsparsity, 3)}")
    

    # StandardNet
    model = NormNet(X_train.shape[1], hidden_size, 1, n_hidden = n_hidden,
                share=False, use_bias=True, use_l2_wn=True, use_1pathnorm=False)
    performance, nsparsity = get_NN_result(model, compute_loss, compute_eval,
                                            dl_train, dl_val, dl_test,
                                            lambda_list,
                                            verbose=verbose, 
                                            use_es=use_es,
                                            compile=compile)
    standard_performances[dataset.name] = performance
    standard_nsparsities[dataset.name] = nsparsity
    print(f"\nFinished Standard Net Experiment: Performance: {np.round(performance, 5)}")
    print(f"Finished Standard Net Experiment: NS: {np.round(nsparsity, 3)}")
    