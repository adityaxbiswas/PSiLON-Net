

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler
from torch import optim
import lightning as L
from itertools import product

def convert_to_dataloader(X, Y, to_numpy=False, batch_size=None, 
                          shuffle=False, accelerator='cpu',
                          num_workers=0):
    if to_numpy:
        X = X.to_numpy()
        Y = Y.to_numpy()
    X = torch.from_numpy(X).to(torch.float32)
    Y = torch.from_numpy(Y).to(torch.float32)
    if accelerator == 'gpu':
        X = X.to(device='cuda')
        Y = Y.to(device='cuda')
    if batch_size is None:
        batch_size = len(X)
    dl =  DataLoader(TensorDataset(X,Y), 
                      batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)
    return dl

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat,y))
    
class MulticlassAccuracy(nn.Module):
    def __init__(self, top_k=1):
        super().__init__()
        self.top_k = top_k
    def forward(self, y_hat_logits, y):
        _, y_hat_top_k = torch.topk(y_hat_logits, k=self.top_k, dim=1)
        matches = torch.any(y_hat_top_k == y.unsqueeze(1), dim = 1)
        return torch.sum(matches)/y_hat_logits.shape[0]

class LitNetwork(L.LightningModule):
    def __init__(self, model, lambda_, compute_loss, compute_eval,
                  n_batches, n_epochs, compute_test=None, lr=1e-3, output_size=1):
        super().__init__()
        self.model = model
        self.compute_loss = compute_loss
        self.compute_eval = compute_eval
        self.compute_test = compute_eval if compute_test is None else compute_test
        self.lambda_ = lambda_
        self.n_batches = n_batches
        self.n_epochs = n_epochs
        self.lr = lr
        self.output_size = output_size

    def forward(self, X):
        return self.model(X)
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                                lr = self.lr,
                                weight_decay = 0)
        sch = self.build_scheduler(optimizer)
        return [optimizer], [sch]
    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y_hat = self.model(X)
        if self.output_size == 1:
            y_hat = y_hat[:,0]
        loss = self.compute_loss(y_hat, y)
        loss_reg = loss + self.lambda_*self.model.compute_reg()
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_loss_reg', loss_reg, on_step=False, on_epoch=True)
        return loss_reg
    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        y_hat = self.model(X)
        if self.output_size == 1:
            y_hat = y_hat[:,0]
        loss = self.compute_eval(y_hat, y)
        nsparsity = self.get_sparsity(near=True)
        self.log('val_loss', loss)
        self.log('near sparsity', nsparsity)
    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        y_hat = self.model(X)
        if self.output_size == 1:
            y_hat = y_hat[:,0]
        loss = self.compute_test(y_hat, y)
        self.log('test_loss', loss)
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)
    def build_scheduler(self, optimizer):
        factor = self.lr/1e-4
        scheduler = lr_scheduler.OneCycleLR(optimizer,
                                            max_lr=self.lr,
                                            total_steps = self.n_epochs,
                                            pct_start = 0.2,
                                            div_factor = factor,
                                            final_div_factor = factor/10)
        return scheduler
    def get_sparsity(self, near=True):
        return self.model.get_sparsity(near)


class HyperparameterRecorder():
    """
    Take in a dictionary with parameter names as keys and possible settings for
    it as values in the form of a list. It will iterate through every combination.
    Usage: Create a while loop using is_complete.  Within, 
        alternate btw calling next and record to get the current iterations
        parameter dictionary and recording the performance results of the run.
        At the end, call get_best to get best parameter settings
    Note: Designed so that smaller performance/loss is considered better
    """
    def __init__(self, param_dict, verbose = False):
        self.params = [dict(zip(param_dict.keys(),item)) 
                        for item in product(*param_dict.values())]
        self.n_settings = len(self.params)
        self.counter = 0
        self.needs_record = False
        self.performances = [1e8 for _ in range(self.n_settings)]
        self.verbose = verbose
    def next(self):
        if not self.is_complete():
            if not self.needs_record:
                if self.verbose:
                    print(f"\nNow trying: {self.params[self.counter]}")
                self.needs_record = True
                return self.params[self.counter]
            else:
                raise Exception("Previous results were not recorded.")
        else:
            raise Exception("No remaining parameter combinations.")
    def is_complete(self):
        return False if self.counter < self.n_settings else True
    def record(self, performance):
        if self.needs_record:
            self.performances[self.counter] = performance
            self.needs_record = False
            self.counter += 1
            if self.verbose:
                idx = np.argmin(self.performances)
                print(f"Current performance: {np.round(performance,4)}")
                print(f"Best so far: {np.round(self.performances[idx],4)}, {self.params[idx]}")
        else:
            raise Exception("No associated parameters or result already recorded.")
    def get_best(self):
        if self.is_complete():
            idx = np.argmin(self.performances)
            return self.params[idx]
        else:
            raise Exception("Loop incomplete.")
    def get_performances(self):
        return list(zip(self.performances, self.params))


# Not used. Switched to using pytorch lightning so not needed
class Printer(object):
    """
    Printing-to-console tool for usage with neural network training.
    Usage: call update_train/update_eval when you have a new loss value
    and call step at the end of a batch. Will choose to print when appropriate
    based on the initialization settings.
    """
    def __init__(self, window_length = 50, print_every_n_epochs = 100, 
                 batches_per_epoch = 1, running_init = 1):
        self.running_loss_train = running_init
        self.running_loss_eval = running_init
        self.window_length = window_length # exponential moving average window
        self.print_every = print_every_n_epochs
        self.batch_counter = 0
        self.batches_per_epoch = batches_per_epoch
        self.total_epochs_passed = 0

    def calc_print_str(self, running_loss, eval = False):
        part1 = f"Epoch: {self.total_epochs_passed+1}  |  "
        part2 = "Running Loss"
        if eval:
            part2 += " Eval"
        part2 += f": {round(running_loss, 6)}"
        return part1 + part2
    def print(self, running_loss, eval = False):
        if ((self.total_epochs_passed+1) % self.print_every == 0) & \
               ((self.batch_counter+1) == self.batches_per_epoch):
                print(self.calc_print_str(running_loss, eval))
    def calc_update(self, running_loss, loss):
        return (running_loss*(self.window_length-1) + loss)/self.window_length
    def update_train(self, loss):
        self.running_loss_train = self.calc_update(self.running_loss_train, loss.item())
        self.print(self.running_loss_train)
    def update_eval(self, loss):
        self.running_loss_eval = self.calc_update(self.running_loss_eval, loss.item())
        self.print(self.running_loss_eval, True)
    def step(self):
        self.batch_counter += 1
        if self.batch_counter == self.batches_per_epoch:
            self.batch_counter = 0
            self.total_epochs_passed += 1

