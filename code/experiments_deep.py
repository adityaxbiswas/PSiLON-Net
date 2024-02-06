
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


import torchvision
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
seed = 5392868
torch.manual_seed(seed)
#torch.use_deterministic_algorithms(True)
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)

import logging
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

import math
from NN_utils import *
from NeuralNetwork import *

# master nn training function
def get_NN_result(model, compute_loss, compute_eval, compute_test,
                   dl_train, dl_val, dl_test,
                   lambda_list, verbose=True,
                    compile=True, lr=1e-3, output_size=1,
                    accelerator='cpu'):
    
    recorder = HyperparameterRecorder({'lambda_': lambda_list}, 
                                    verbose = verbose)
    while not recorder.is_complete():
        hyperparams = recorder.next()
        lambda_ = hyperparams['lambda_']

        # create lightning model
        if compile:
            model = torch.jit.script(model)
        litmodel = LitNetwork(model, lambda_=lambda_,
                            compute_loss=compute_loss,
                            compute_eval=compute_eval,
                            compute_test=compute_test,
                            n_batches = n_batches,
                            n_epochs = epochs,
                            lr=lr,
                            output_size=output_size)
        
        # define trainer
        mc_dir = path.join(checkpoint_folder, f"model-reg-{lambda_}")
        check_val_every_n_epoch = 5
        trainer = L.Trainer(callbacks=[], 
                            max_epochs = epochs,
                            accelerator=accelerator,
                            num_sanity_val_steps=0,
                            log_every_n_steps=5,
                            check_val_every_n_epoch = check_val_every_n_epoch,
                            enable_checkpointing=True,
                            enable_progress_bar=True,
                            enable_model_summary=False)
        
        # train, save model, predict, and record
        trainer.fit(litmodel, dl_train, dl_val)
        trainer.save_checkpoint(path.join(mc_dir, 'model_final.ckpt'))
        performance = trainer.validate(litmodel, dl_val, verbose = False)[0]['val_loss']
        if verbose:
            test_performance = trainer.test(litmodel, dl_test, verbose = False)[0]['test_loss']
            print(f"Test Accuracy: {np.round(test_performance, 5)}")
        recorder.record(performance)

    # load best model and predict on test set
    hyperparams = recorder.get_best()
    lambda_ = hyperparams['lambda_']
    print(f"Optimal value of Lambda: {lambda_}")
    mc_dir = path.join(checkpoint_folder, f"model-reg-{lambda_}")
    final_model_path = path.join(mc_dir, 'model_final.ckpt')
    litmodel = LitNetwork.load_from_checkpoint(final_model_path,
                                            model = model,
                                            lambda_=lambda_,
                                            compute_loss=compute_loss,
                                            compute_eval=compute_eval,
                                            compute_test=compute_test,
                                            n_batches = n_batches,
                                            n_epochs = epochs,
                                            lr=lr,
                                            output_size=output_size)
    performance = trainer.test(litmodel, dl_test, verbose = False)[0]['test_loss']
    cross_entropy = trainer.validate(litmodel, dl_test, verbose = False)[0]['val_loss']
    nsparsity = litmodel.get_sparsity(near=True)

    return performance, cross_entropy, nsparsity


##########################################################################################

# load and preprocess all data
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    torch.nn.Flatten(0)
])
data_folder = path.join(project_folder, "data", "fashion")
data_train = torchvision.datasets.FashionMNIST(data_folder,
                                               train=True,
                                               download=True,
                                               transform=input_transform)
data_test = torchvision.datasets.FashionMNIST(data_folder,
                                              train=False,
                                              download=True,
                                              transform=input_transform)

# randomly subset into train/val
n_keep_train = 10000
n_keep_val = 20000
data_train, data_val = torch.utils.data.random_split(data_train, [60000-n_keep_val,
                                                                  n_keep_val])
data_train, _ = torch.utils.data.random_split(data_train, [n_keep_train, 
                                                           60000-n_keep_val-n_keep_train])


# create data loaders
batch_size = 200
n_batches = int(math.ceil(n_keep_train/batch_size))
dl_train = DataLoader(data_train, batch_size=batch_size, shuffle=True,
                         num_workers=0, pin_memory=True)
dl_val = DataLoader(data_val, batch_size=2000,
                         num_workers=0, pin_memory=True)
dl_test = DataLoader(data_test, batch_size=2000,
                         num_workers=0)


# model seettings
compute_loss = nn.CrossEntropyLoss()
compute_eval = nn.CrossEntropyLoss()
compute_test = MulticlassAccuracy(top_k=1)
epochs = 400
input_size = 28*28
hidden_size = 500
output_size = 10
n_hidden = 10
verbose = True
accelerator = 'gpu'
compile = True
lr = 2e-2


# lambda_list = [1e-3, 2.5e-3, 5e-3, 1e-2, 2.5e-2, 5e-2, 1e-1] # used for standard net
# lambda_list = [1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3] # used for all others

#PSiLONNet
lambda_list = [2.5e-4] # best
model = NormResNet(input_size, hidden_size, output_size, 
                   n_hidden = n_hidden, share=True, use_bias=True, 
                   use_l2_wn=False, use_1pathnorm=True,
                   use_improved_1pathnorm = True)
performance, ce, nsparsity = get_NN_result(model, compute_loss, compute_eval,
                                        compute_test,
                                        dl_train, dl_val, dl_test,
                                        lambda_list,
                                        verbose=verbose,
                                        compile=compile,
                                        lr=lr,
                                        output_size=output_size,
                                        accelerator=accelerator)


# L1-IB
lambda_list = [2.5e-4] # best
model = NormResNet(input_size, hidden_size, output_size, 
                   n_hidden = n_hidden, share=False, use_bias=True, 
                   use_l2_wn=False, use_1pathnorm=True,
                   use_improved_1pathnorm = True)
performance, ce, nsparsity = get_NN_result(model, compute_loss, compute_eval,
                                        compute_test,
                                        dl_train, dl_val, dl_test,
                                        lambda_list,
                                        verbose=verbose,
                                        compile=compile,
                                        lr=lr,
                                        output_size=output_size,
                                        accelerator=accelerator)


# L2-IB
lambda_list = [2.5e-4] # best
model = NormResNet(input_size, hidden_size, output_size, 
                   n_hidden = n_hidden, share=False, use_bias=True, 
                   use_l2_wn=True, use_1pathnorm=True,
                   use_improved_1pathnorm = True)
performance, ce, nsparsity = get_NN_result(model, compute_loss, compute_eval,
                                        compute_test,
                                        dl_train, dl_val, dl_test,
                                        lambda_list,
                                        verbose=verbose,
                                        compile=compile,
                                        lr=lr,
                                        output_size=output_size,
                                        accelerator=accelerator)

# L1-OB
lambda_list = [1e-4] # best
model = NormResNet(input_size, hidden_size, output_size, 
                   n_hidden = n_hidden, share=False, use_bias=True, 
                   use_l2_wn=False, use_1pathnorm=True,
                   use_improved_1pathnorm = False)
performance, ce, nsparsity = get_NN_result(model, compute_loss, compute_eval,
                                        compute_test,
                                        dl_train, dl_val, dl_test,
                                        lambda_list,
                                        verbose=verbose,
                                        compile=compile,
                                        lr=lr,
                                        output_size=output_size,
                                        accelerator=accelerator)



# L2-OB
lambda_list = [1e-4] # best
model = NormResNet(input_size, hidden_size, output_size, 
                   n_hidden = n_hidden, share=False, use_bias=True, 
                   use_l2_wn=True, use_1pathnorm=True,
                   use_improved_1pathnorm = False)
performance, ce, nsparsity = get_NN_result(model, compute_loss, compute_eval,
                                        compute_test,
                                        dl_train, dl_val, dl_test,
                                        lambda_list,
                                        verbose=verbose,
                                        compile=compile,
                                        lr=lr,
                                        output_size=output_size,
                                        accelerator=accelerator)



# standard net
lambda_list = [2.5e-2] # best
model = NormResNet(input_size, hidden_size, output_size, 
                   n_hidden = n_hidden, share=False, use_bias=True, 
                   use_l2_wn=True, use_1pathnorm=False,
                   use_improved_1pathnorm = False)
performance, ce, nsparsity = get_NN_result(model, compute_loss, compute_eval,
                                        compute_test,
                                        dl_train, dl_val, dl_test,
                                        lambda_list,
                                        verbose=verbose,
                                        compile=compile,
                                        lr=lr,
                                        output_size=output_size,
                                        accelerator=accelerator)


# Projected gradient descent PSiLON
# to do this, at the start of the forward pass in CReLUNormLinear and NormLinear
# temporarily add a step, where we perform the projection in a torch.no_grad() context:
# you will get what you need from parameters using .data, manipulate it, and copy
# it back to the parameter using .copy_ 
# this will need to be performed within a new python session, since it will require
# a recompile of the network, since we are using torch.jit
lambda_list = [2.5e-4]
model = NormResNet(input_size, hidden_size, output_size, 
                   n_hidden = n_hidden, share=True, use_bias=True, 
                   use_l2_wn=False, use_1pathnorm=True,
                   use_improved_1pathnorm = True)
performance, ce, nsparsity = get_NN_result(model, compute_loss, compute_eval,
                                        compute_test,
                                        dl_train, dl_val, dl_test,
                                        lambda_list,
                                        verbose=verbose,
                                        compile=compile,
                                        lr=lr,
                                        output_size=output_size,
                                        accelerator=accelerator)
