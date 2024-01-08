

import math
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler




def convert_to_dataloader(X, Y, to_numpy = False, batch_size = None, shuffle = False):
    if to_numpy:
        X = X.to_numpy()
        Y = Y.to_numpy()
    X = torch.from_numpy(X).to(torch.float32)
    Y = torch.from_numpy(X).to(torch.float32)
    if batch_size is None:
        batch_size = len(X)
    return DataLoader(TensorDataset(X,Y), batch_size = batch_size, shuffle = shuffle)

# Usage example
#batch_size = X_train.shape[0]
#batches_per_epoch = int(math.ceil(X_train.shape[0]/batch_size))
#dataloader_train = convert_to_dataloader(X_train, Y_train, to_numpy = True, batch_size = batch_size)
#dataloader_val = convert_to_dataloader(X_val, Y_val, to_numpy = True)
#dataloader_test = convert_to_dataloader(X_test, Y_test, to_numpy = True)



def build_scheduler(optimizer, milestones):
    scheduler1 = lr_scheduler.LinearLR(optimizer, 
                                       start_factor = 1e-4,
                                       end_factor = 1, 
                                       total_iters = milestones[0])
    scheduler2 = lr_scheduler.LinearLR(optimizer, 
                                       start_factor = 1,
                                       end_factor = 1e-2, 
                                       total_iters = milestones[1])
    scheduler3 = lr_scheduler.MultiplicativeLR(optimizer, 
                                              lr_lambda = lambda epoch: 0.999,
                                              verbose = False)
    scheduler = lr_scheduler.SequentialLR(optimizer, 
                                          [scheduler1, scheduler2, scheduler3],
                                          milestones = milestones)
    return scheduler


class Printer(object):
    def __init__(self, window_length = 50, print_every_n_epochs = 100, 
                 batches_per_epoch = 1, running_init = 1):
        self.running_loss_train = running_init
        self.running_loss_eval = running_init
        self.window_length = window_length
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




class HyperparameterRecorder():
    def __init__(self, param_dict, verbose = False):
        self.params = [dict(zip(param_dict.keys(),item)) 
                        for item in product(*param_dict.values())]
        self.n_settings = len(self.params)
        self.counter = 0
        self.needs_record = False
        self.performances = [-1e8 for _ in range(self.n_settings)]
        self.verbose = verbose
    def next(self):
        if not self.is_complete():
            if not self.needs_record:
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
                idx = np.argmax(self.performances)
                print(f"Best so far: {np.round(self.performances[idx],4)}, {self.params[idx]}")
        else:
            raise Exception("No associated parameters or result already recorded.")
    def get_best(self):
        if self.is_complete():
            return self.params[self.best_idx]
        else:
            raise Exception("Loop incomplete.")
    def get_performances(self):
        return list(zip(self.performances, self.params))