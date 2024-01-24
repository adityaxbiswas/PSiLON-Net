
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
from torch.nn import functional as F
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import math
from NN_utils import *
from NeuralNetwork import *

def get_NN_result(model, compute_loss, compute_eval,
                   dl_train, dl_val, dl_test,
                   lambda_list, verbose=True,
                    compile=True, lr=1e-3):
    
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
                            lr=lr)
        
        # define trainer
        mc_dir = path.join(checkpoint_folder, f"model-reg-{lambda_}")
        check_val_every_n_epoch = 1
        trainer = L.Trainer(callbacks=[], 
                            max_epochs = epochs,
                            accelerator=accelerator,
                            num_sanity_val_steps=0,
                            log_every_n_steps=5,
                            check_val_every_n_epoch = check_val_every_n_epoch,
                            enable_checkpointing=True,
                            enable_progress_bar=False,
                            enable_model_summary=False)
        
        # train, predict on val, and record
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
    #clear_folder(lightning_logs_folder)
    #clear_folder(checkpoint_folder)
    return performance, nsparsity

project_folder = path.abspath(path.join(path.sep,"Projects", "PSiLON-Net"))
data_folder = path.join(project_folder, "data", "fashion")

input_transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
data_train = torchvision.datasets.FashionMNIST(data_folder,
                                               train=True,
                                               download=True,
                                               transform=input_transform)
data_test = torchvision.datasets.FashionMNIST(data_folder,
                                              train=False,
                                              download=True,
                                              transform=input_transform)
data_train, data_val = torch.utils.data.random_split(data_train, [50000, 10000])



batch_size = 100
n_batches = int(math.ceil(50000/batch_size))
train_loader = DataLoader(data_train, batch_size=100, shuffle=True,
                         num_workers=0, pin_memory=True)
val_loader = DataLoader(data_val, batch_size=500,
                         num_workers=0, pin_memory=True)
test_loader = DataLoader(data_test, batch_size=500,
                         num_workers=0)


compute_loss = nn.CrossEntropyLoss()
compute_eval = MulticlassAccuracy()
epochs = 100
input_size = 28*28
hidden_size = 500
output_size = 10
n_hidden = 2
verbose = True
accelerator = 'gpu'
compile = True


lambda_list = [1e-5]


#PSiLONNet
model = NormResNet(input_size, hidden_size, output_size, 
                   n_hidden = n_hidden, share=True, use_bias=True, 
                   use_l2_wn=False, use_1pathnorm=True,
                   use_improved_1pathnorm = True)
performance, nsparsity = get_NN_result(model, compute_loss, compute_eval,
                                        train_loader, val_loader, test_loader,
                                        lambda_list,
                                        verbose=verbose,
                                        compile=compile,
                                        lr=1e-3)