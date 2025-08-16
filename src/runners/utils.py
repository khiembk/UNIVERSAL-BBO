import os
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import numpy as np
import design_bench

def remove_file(fpath):
    if os.path.exists(fpath):
        os.remove(fpath)


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)
    return dir


def make_save_dirs(args, prefix, suffix=None, with_time=False):
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S") if with_time else ""
    suffix = suffix if suffix is not None else ""

    result_path = make_dir(os.path.join(args.result_path, prefix, suffix, time_str))
    checkpoint_path = make_dir(os.path.join(result_path, "checkpoint"))

    return checkpoint_path


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Parameter') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_optimizer(optim_config, parameters):
    if optim_config.optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay,
                                betas=(optim_config.beta1, 0.999))
    elif optim_config.optimizer == 'RMSProp':
        return torch.optim.RMSprop(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay)
    elif optim_config.optimizer == 'SGD':
        return torch.optim.SGD(parameters, lr=optim_config.lr, momentum=0.9)
    else:
        return NotImplementedError('Optimizer {} not understood.'.format(optim_config.optimizer))

### Sampling data from GP model
def sampling_data_from_GP(x_train, device, GP_Model, num_gradient_steps = 50, num_functions = 5, num_points = 10, learning_rate = 0.001, delta_lengthscale = 0.1, delta_variance = 0.1, seed = 0, threshold_diff = 0.1):
    lengthscale = GP_Model.kernel.lengthscale
    variance = GP_Model.variance 
    torch.manual_seed(seed=seed)
    datasets={}
    learning_rate_vec = torch.cat((-learning_rate*torch.ones(num_points, x_train.shape[1], device=device), learning_rate*torch.ones(num_points, x_train.shape[1], device = device)))


    for iter in range(num_functions):
        datasets[f'f{iter}']=[]
        
        new_lengthscale = lengthscale + delta_lengthscale*(torch.rand(1, device=device)*2 -1)
        new_variance = variance + delta_variance*(torch.rand(1, device=device)*2 -1)
        
        GP_Model.set_hyper(lengthscale=new_lengthscale,variance = new_variance)
    
        selected_indices = torch.randperm(x_train.shape[0])[:num_points]
        low_x = x_train[selected_indices].clone().detach().requires_grad_(True)
        high_x = x_train[selected_indices].clone().detach().requires_grad_(True)
        joint_x = torch.cat((low_x, high_x)) 
        
        # Using gradient ascent and descent to find high and low designs 
        for t in range(num_gradient_steps): 
            mu_star = GP_Model.mean_posterior(joint_x)
            grad = torch.autograd.grad(mu_star.sum(),joint_x)[0]
            joint_x += learning_rate_vec*grad 

        
        joint_y = GP_Model.mean_posterior(joint_x)
        
        low_x = joint_x[:num_points,:]
        high_x = joint_x[num_points:,:]
        low_y = joint_y[:num_points]
        high_y = joint_y[num_points:]
        
        
        for i in range(num_points):
            if high_y[i] - low_y[i] <= threshold_diff:
                continue
            sample = [(high_x[i].detach(),high_y[i].detach()),(low_x[i].detach(),low_y[i].detach())]
            datasets[f'f{iter}'].append(sample)

    # restore lengthscale and variance of GP
    GP_Model.kernel.lengthscale = lengthscale
    GP_Model.variance = variance
    
    return datasets

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        [[x_high, y_high], [x_low, y_low]] = self.data[idx]
        return (x_high, y_high), (x_low, y_low)

# Create a DataLoader for each epoch
def create_train_dataloader(data_from_GP, val_frac=0.2, batch_size=32, shuffle=True):
    train_data = []
    val_data = []
    for function, function_samples in data_from_GP.items():
        train_data = train_data + function_samples[int(len(function_samples)*val_frac):]
        val_data = val_data + function_samples[:int(len(function_samples)*val_frac)]
        
    train_dataset = CustomDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, val_data

def create_val_dataloader(val_dataset, batch_size=32, shuffle=False):
    
    valid_dataset = CustomDataset(val_dataset)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)

    return valid_dataloader

### Sampling 128 designs from offline data
def sampling_from_offline_data(x, y, n_candidates=128, type='random', percentile_sampling=0.2, seed=0):
    y = y.view(-1)
    indices = torch.argsort(y)
    x = x[indices]
    y = y[indices]
    if type == 'highest':
        print('highest')
        return x[-n_candidates:], y[-n_candidates:] 
    if type == 'lowest': 
        return x[:n_candidates], y[:n_candidates:]
    tmp = len(x)
    percentile_index = int(percentile_sampling * len(x))
    if type == 'low':
        x = x[:percentile_index]
        y = y[:percentile_index]
    if type == 'high':
        x = x[tmp-percentile_index:]
        y = y[tmp-percentile_index:]
    np.random.seed(seed)
    indices = np.random.choice(x.shape[0], size = n_candidates, replace=False)
    return x[indices], y[indices]
    
### Sampling 128 designs from offline data

### Testing 128 found designs by the oracle
def testing_by_oracle(task_name, high_candidates):
    if task_name != 'TFBind10-Exact-v0':
        task = design_bench.make(task_name)
    else:
        task = design_bench.make(task_name,
                                dataset_kwargs={"max_samples": 10000})
    high_candidates = high_candidates.numpy()
    score = task.predict(high_candidates)
    return score
### Testing 128 found designs by the oracle
