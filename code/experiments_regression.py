

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
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor as RFR

from torch import nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning as L
import logging
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

# import custom functions/classes
from NN_utils import HyperparameterRecorder, convert_to_dataloader
from NeuralNetwork import *
###################################################################################
### Main neural network training function
def get_NN_result(Net, compute_loss, compute_eval,
                   dl_train, dl_val, dl_test,
                   lambda_list, verbose=True, use_es=True,
                     residual=False, accelerator="cpu"):
    # [1e-3,3.3e-3,1e-2,3.3e-2,1e-1]
    recorder = HyperparameterRecorder({'lambda_': lambda_list}, 
                                    verbose = verbose)
    mc_dict = {}
    while not recorder.is_complete():
        # create lightning model
        hyperparams = recorder.next()
        lambda_ = hyperparams['lambda_']
        h = int(hidden_size/2) if residual else hidden_size
        model = torch.jit.script(Net(X_train.shape[1], h, 1,
                                      n_hidden = n_hidden,
                                      accelerator = accelerator))
        litmodel = LitNetwork(model, lambda_=lambda_,
                            compute_loss=compute_loss,
                            compute_eval=compute_eval,
                            total_steps = epochs*n_batches)
        
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
                                            compute_loss=nn.MSELoss(),
                                            compute_eval=RMSELoss(),
                                            total_steps = epochs*n_batches)
    performance = trainer.test(litmodel, dl_test, verbose = False)[0]['test_loss']

    # clear checkpoint and logs folders
    clear_folder(lightning_logs_folder)
    clear_folder(checkpoint_folder)
    return performance
###################################################################################
# smaller helper function/tools
seed = 980679325
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)

RMSE = lambda Y,Y_hat: np.sqrt(np.mean((Y-Y_hat)**2))
def select_continuous_features(feature_names, categorical_indicator):
    features_categorical = itertools.compress(feature_names, categorical_indicator)
    return list(set(feature_names) - set(features_categorical))

repeat_dataset_names = ['delays_zurich_transport', 'diamonds', 'Brazilian_houses',
                        'Bike_Sharing_Demand', 'nyc-taxi-green-dec-2016',
                        'house_sales', 'medical_charges', 'abalone']

######################################################################################
### START SCRIPT


rf_performances = {}
lr_performances = {}
psilon_performances = {}
standard_performances = {}

n_keep = 1000
#SUITE_ID = 335 # Regression on numerical and categorical features
SUITE_ID = 336 # Regression on numerical features
#SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite
for task_id in benchmark_suite.tasks:  # iterate over all tasks
    # download/load the dataset
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    if SUITE_ID == 336:
        if dataset.name in repeat_dataset_names:
            continue

    X, y, categorical_indicator, feature_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    if len(X.columns) < 10:
        continue
    print("Loaded dataset: " + dataset.name)

    # randomly reduce it in size and train/test split
    n = min(len(y),n_keep)
    shuffler = np.random.default_rng(seed=seed).permutation(n)
    X, y = X.iloc[shuffler,:], y.iloc[shuffler]
    X = X.iloc[:n,:]
    y = y.iloc[:n]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.4, 
                                                        shuffle=False)

    # preprocess the data and split into train/val
    features_continuous = select_continuous_features(feature_names, categorical_indicator)
    transformer = ColumnTransformer([('scaler', StandardScaler(), features_continuous)],
                                    remainder = 'passthrough')
    X_train = transformer.fit_transform(X_train).astype(np.float32)
    X_test = transformer.transform(X_test).astype(np.float32)
    y_median, y_qd = y.median(), iqr(y)/2
    y_train = (y_train-y_median)/y_qd
    y_test = (y_test-y_median)/y_qd
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=2/3,
                                                      shuffle=False)

    ################################################################################
    # Neural Network training

    compute_loss = nn.MSELoss()
    compute_eval = RMSELoss()
    epochs = 2500
    use_es = False # es = early stopping
    hidden_size = 500
    n_hidden = 3
    verbose = True
    accelerator = 'gpu'
    n_batches = 4
    batch_size = int(math.ceil(len(X_train)/n_batches))

    dl_train = convert_to_dataloader(X_train, y_train.values,
                                     batch_size=batch_size,
                                     accelerator='cpu')
    dl_val = convert_to_dataloader(X_val, y_val.values,
                                     accelerator='cpu')
    dl_test = convert_to_dataloader(X_test, y_test.values,
                                     accelerator='cpu')

    
    #PSiLONNet
    # [1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2, 2.5e-2]
    lambda_list = [5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2, 
                   2.5e-2, 5e-2, 1e-1, 2.5e-1, 5e-1]
    performance = get_NN_result(PSiLONNet, compute_loss, compute_eval,
                                dl_train, dl_val, dl_test,
                                lambda_list,
                                verbose =  verbose, 
                                use_es = use_es, residual = False,
                                accelerator = accelerator)
    psilon_performances[dataset.name] = performance
    print(f"Finished PSiLON Net Experiment: {np.round(performance, 5)}")
    
    

    # StandardNet
    lambda_list = [1e-3, 2.5e-3, 5e-3, 1e-2, 
                   2.5e-2, 5e-2, 1e-1, 2.5e-1]
    performance = get_NN_result(StandardNet, compute_loss, compute_eval,
                                dl_train, dl_val, dl_test,
                                lambda_list,
                                verbose = verbose, 
                                use_es = use_es, residual = False,
                                accelerator = accelerator)
    standard_performances[dataset.name] = performance
    print(f"Finished Standard-Net Experiment: {np.round(performance, 5)}")



    ######################################################################################
    ### Baseline model training

    '''
    # Ridge Regression
    recorder = HyperparameterRecorder({'alpha': [1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,
                                                 1e-3,3e-3,1e-2,3e-2,1e-1,0.3,1,3,10,30,100]},
                                        verbose = False)
    while not recorder.is_complete():
        hyperparams = recorder.next()
        model = Ridge(alpha = hyperparams['alpha'])
        model.fit(X_train, y_train)
        y_hat_val = model.predict(X_val)
        recorder.record(RMSE(y_val, y_hat_val))
    hyperparams = recorder.get_best()
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
        model = RFR(n_estimators = 1000, 
                    min_samples_leaf = hyperparams['min_samples_leaf'],
                    n_jobs = 38)
        model.fit(X_train, y_train)
        y_hat_val = model.predict(X_val)
        recorder.record(RMSE(y_val, y_hat_val))
    hyperparams = recorder.get_best()
    model = RFR(n_estimators = 1000, 
                min_samples_leaf = hyperparams['min_samples_leaf'],
                n_jobs = 38)
    model.fit(X_train, y_train)
    y_hat_test = model.predict(X_test)
    performance = RMSE(y_test, y_hat_test)
    rf_performances[dataset.name] = performance
    print(f"Finished Random Forest Experiment: {np.round(performance, 5)}")
    '''