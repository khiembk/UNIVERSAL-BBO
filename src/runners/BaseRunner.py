import argparse
import datetime
import time

import yaml
import os
import traceback

import torch

from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm

from runners.base.EMA import EMA
from runners.utils import make_save_dirs, remove_file, sampling_data_from_GP, create_train_dataloader, create_val_dataloader, sampling_from_offline_data, testing_by_oracle
import numpy as np

import gpytorch 
from gaussian_process.GPlib import ExactGPModel
from gaussian_process.GP import GP
import design_bench

class BaseRunner(ABC):
    def __init__(self, config):
        self.net = None  # Neural Network
        self.optimizer = None  # optimizer
        self.scheduler = None  # scheduler
        self.config = config  # config from configuration file
        # set training params
        self.global_epoch = 0  # global epoch
        self.global_step = 0

        self.GAN_buffer = {}  # GAN buffer for Generative Adversarial Network
        self.topk_checkpoints = {}  # Top K checkpoints

        # set log and save destination
        self.config.result = argparse.Namespace()
        if self.config.args.train:
            self.config.result.ckpt_path = make_save_dirs(self.config.args,
                                                    prefix=self.config.task.name + f'/seed{self.config.args.seed}',
                                                    suffix=self.config.model.model_name,
                                                    with_time=False)

            self.save_config()  # save configuration file
            
        # initialize model
        self.net, self.optimizer, self.scheduler = self.initialize_model_optimizer_scheduler(self.config)


        # initialize EMA
        self.use_ema = False if not self.config.model.__contains__('EMA') else self.config.model.EMA.use_ema
        if self.use_ema:
            self.ema = EMA(self.config.model.EMA.ema_decay)
            self.update_ema_interval = self.config.model.EMA.update_ema_interval
            self.start_ema_step = self.config.model.EMA.start_ema_step
            self.ema.register(self.net)

        # load model from checkpoint
        self.load_model_from_checkpoint()

        # initialize DDP
        self.net = self.net.to(self.config.training.device[0])

        # get offline data from design-bench
        
        self.offline_x, self.mean_offline_x, self.std_offline_x, self.offline_y, self.mean_offline_y, self.std_offline_y = self.get_offline_data()
        
        if self.config.task.normalize_x:
            self.offline_x = (self.offline_x - self.mean_offline_x) / self.std_offline_x
        if self.config.task.normalize_y:
            self.offline_y = (self.offline_y - self.mean_offline_y) / self.std_offline_y
    
        self.offline_x = self.offline_x.to(self.config.training.device[0])
        self.offline_y = self.offline_y.to(self.config.training.device[0])

    def get_offline_data(self):
        if self.config.task.name != 'TFBind10-Exact-v0':
            task = design_bench.make(self.config.task.name)
        else:
            task = design_bench.make(self.config.task.name,
                                    dataset_kwargs={"max_samples": 10000})

        offline_x = task.x
        if task.is_discrete:
            offline_x = task.to_logits(offline_x).reshape(offline_x.shape[0], -1)

        mean_x = np.mean(offline_x, axis=0)
        std_x = np.std(offline_x, axis=0)
        std_x = np.where(std_x == 0, 1.0, std_x)
        
        offline_y = task.y
        mean_y = np.mean(offline_y, axis=0)
        std_y = np.std(offline_y, axis=0)
        
        shuffle_idx = np.random.permutation(offline_x.shape[0])

        offline_x = offline_x[shuffle_idx]
        offline_y = offline_y[shuffle_idx]
        offline_y = offline_y.reshape(-1)
        
        return torch.from_numpy(offline_x), torch.from_numpy(mean_x), torch.from_numpy(std_x), torch.from_numpy(offline_y), torch.from_numpy(mean_y), torch.from_numpy(std_y)

    # print msg
    def logger(self, msg, **kwargs):
        print(msg, **kwargs)

    # save configuration file
    def save_config(self):
        save_path = os.path.join(self.config.result.ckpt_path, 'config.yaml')
        save_config = self.config
        with open(save_path, 'w') as f:
            yaml.dump(save_config, f)

    def initialize_model_optimizer_scheduler(self, config, is_test=False):
        """
        get model, optimizer, scheduler
        :param args: args
        :param config: config
        :param is_test: is_test
        :return: net: Neural Network, nn.Module;
                 optimizer: a list of optimizers;
                 scheduler: a list of schedulers or None;
        """
        net = self.initialize_model(config)
        optimizer, scheduler = None, None
        if not is_test:
            optimizer, scheduler = self.initialize_optimizer_scheduler(net, config)
        return net, optimizer, scheduler

    # load model, EMA, optimizer, scheduler from checkpoint
    def load_model_from_checkpoint(self):
        model_states = None
        if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
            model_states = torch.load(self.config.model.model_load_path, map_location='cpu')

            self.global_epoch = model_states['epoch']
            self.global_step = model_states['step']

            # load model
            self.net.load_state_dict(model_states['model'])

            # load ema
            if self.use_ema:
                self.ema.shadow = model_states['ema']
                self.ema.reset_device(self.net)

            # load optimizer and scheduler
            if self.config.args.train:
                if self.config.model.__contains__('optim_sche_load_path') and self.config.model.optim_sche_load_path is not None:
                    optimizer_scheduler_states = torch.load(self.config.model.optim_sche_load_path, map_location='cpu')
                    for i in range(len(self.optimizer)):
                        self.optimizer[i].load_state_dict(optimizer_scheduler_states['optimizer'][i])

                    if self.scheduler is not None:
                        for i in range(len(self.optimizer)):
                            self.scheduler[i].load_state_dict(optimizer_scheduler_states['scheduler'][i])
        return model_states

    def get_checkpoint_states(self, stage='epoch_end'):
        optimizer_state = []
        for i in range(len(self.optimizer)):
            optimizer_state.append(self.optimizer[i].state_dict())

        scheduler_state = []
        for i in range(len(self.scheduler)):
            scheduler_state.append(self.scheduler[i].state_dict())

        optimizer_scheduler_states = {
            'optimizer': optimizer_state,
            'scheduler': scheduler_state
        }

        model_states = {
            'step': self.global_step,
        }

        model_states['model'] = self.net.state_dict()

        if stage == 'exception':
            model_states['epoch'] = self.global_epoch
        else:
            model_states['epoch'] = self.global_epoch + 1

        if self.use_ema:
            model_states['ema'] = self.ema.shadow
        return model_states, optimizer_scheduler_states

    # EMA part
    def step_ema(self):
        with_decay = False if self.global_step < self.start_ema_step else True
        self.ema.update(self.net, with_decay=with_decay)

    def apply_ema(self):
        if self.use_ema:
            self.ema.apply_shadow(self.net)

    def restore_ema(self):
        if self.use_ema:
            self.ema.restore(self.net)

    # Evaluation and sample part
    @torch.no_grad()
    def validation_step(self, val_batch, epoch, step):
        self.apply_ema()
        self.net.eval()
        loss = self.loss_fn(net=self.net,
                            batch=val_batch,
                            epoch=epoch,
                            step=step,
                            opt_idx=0,
                            stage='val_step')
        if len(self.optimizer) > 1:
            loss = self.loss_fn(net=self.net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=1,
                                stage='val_step')
        self.restore_ema()

    @torch.no_grad()
    def validation_epoch(self, val_loader, epoch):
        self.apply_ema()
        self.net.eval()

        pbar = tqdm(val_loader, total=len(val_loader), smoothing=0.01, disable=False)
        step = 0
        loss_sum = 0.
        dloss_sum = 0.
        for val_batch in pbar:
            loss = self.loss_fn(net=self.net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=0,
                                stage='val',
                                write=False)
            loss_sum += loss
            if len(self.optimizer) > 1:
                loss = self.loss_fn(net=self.net,
                                    batch=val_batch,
                                    epoch=epoch,
                                    step=step,
                                    opt_idx=1,
                                    stage='val',
                                    write=False)
                dloss_sum += loss
            step += 1
        average_loss = loss_sum / step
        self.restore_ema()
        return average_loss


    # abstract methods
    @abstractmethod
    def print_model_summary(self, net):
        pass

    @abstractmethod
    def initialize_model(self, config):
        """
        initialize model
        :param config: config
        :return: nn.Module
        """
        pass

    @abstractmethod
    def initialize_optimizer_scheduler(self, net, config):
        """
        initialize optimizer and scheduler
        :param net: nn.Module
        :param config: config
        :return: a list of optimizers; a list of schedulers
        """
        pass

    @abstractmethod
    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=False):
        """
        loss function
        :param net: nn.Module
        :param batch: batch
        :param epoch: global epoch
        :param step: global step
        :param opt_idx: optimizer index, default is 0; set it to 1 for GAN discriminator
        :param stage: train, val, test
        :param write: write loss information to SummaryWriter
        :return: a scalar of loss
        """
        pass

    @abstractmethod
    def sample(self, net, low_candidates, low_scores, high_cond_scores):
        """
        sample a single batch
        :param net: nn.Module
        :param batch: batch
        :param sample_path: path to save samples
        :param stage: train, val, test
        :return:
        """
        pass


    def on_save_checkpoint(self, net, train_loader, val_loader, epoch, step):
        """
        additional operations whilst saving checkpoint
        :param net: nn.Module
        :param train_loader: train data loader
        :param val_loader: val data loader
        :param epoch: epoch
        :param step: step
        :return:
        """
        pass

    def train(self):
        
        start_epoch = self.global_epoch
        
        try:
            # initialize params for GP
            lengthscale = torch.tensor(self.config.GP.initial_lengthscale, device=self.config.training.device[0])
            variance = torch.tensor(self.config.GP.initial_outputscale, device=self.config.training.device[0])
            noise = torch.tensor(self.config.GP.noise, device=self.config.training.device[0])
            mean_prior = torch.tensor(0.0, device = self.config.training.device[0]) 
            
            if self.config.GP.type_of_initial_points == 'highest':
                best_indices = torch.argsort(self.offline_y)[-1024:]
                self.best_x = self.offline_x[best_indices]
            elif self.config.GP.type_of_initial_points == 'lowest': 
                best_indices = torch.argsort(self.offline_y)[:1024]
                self.best_x = self.offline_x[best_indices]
            else : 
                self.best_x = self.offline_x 
            
            val_loader = None
            val_dataset = []
            
            accumulate_grad_batches = self.config.training.accumulate_grad_batches 
            for epoch in range(start_epoch, self.config.training.n_epochs):
                ### generate data from GP and create dataloader
                start_time = time.time()
                if self.config.task.name == 'TFBind8-Exact-v0': 
                    selected_fit_samples = torch.randperm(self.offline_x.shape[0])[:self.config.GP.num_fit_samples]
                    GP_Model = GP(device=self.config.training.device[0],
                                x_train=self.offline_x[selected_fit_samples],
                                y_train=self.offline_y[selected_fit_samples], 
                                lengthscale=lengthscale, 
                                variance=variance, 
                                noise=noise, 
                                mean_prior=mean_prior)
                else: 
                    GP_Model = GP(device=self.config.training.device[0],
                                x_train=self.offline_x,
                                y_train=self.offline_y, 
                                lengthscale=lengthscale, 
                                variance=variance, 
                                noise=noise, 
                                mean_prior=mean_prior)
                
                data_from_GP = sampling_data_from_GP(x_train=self.best_x,
                                                    device=self.config.training.device[0],
                                                    GP_Model=GP_Model,
                                                    num_functions=self.config.GP.num_functions,
                                                    num_gradient_steps=self.config.GP.num_gradient_steps,
                                                    num_points=self.config.GP.num_points,
                                                    learning_rate=self.config.GP.sampling_from_GP_lr,
                                                    delta_lengthscale=self.config.GP.delta_lengthscale,
                                                    delta_variance=self.config.GP.delta_variance,
                                                    seed=epoch,
                                                    threshold_diff=self.config.GP.threshold_diff)
                train_loader, current_epoch_val_dataset = create_train_dataloader(data_from_GP=data_from_GP,
                                                        val_frac=self.config.training.val_frac,
                                                        batch_size=self.config.training.batch_size,
                                                        shuffle=True)
                val_dataset = val_dataset + current_epoch_val_dataset
                
                pbar = tqdm(train_loader, total=len(train_loader), smoothing=0.01, disable=False)
                self.global_epoch = epoch
                for train_batch in pbar:
                    self.global_step += 1
                    self.net.train()

                    losses = []
                    for i in range(len(self.optimizer)):
                        loss = self.loss_fn(net=self.net,
                                            batch=train_batch,
                                            epoch=epoch,
                                            step=self.global_step,
                                            opt_idx=i,
                                            stage='train')

                        loss.backward()
                        if self.global_step % accumulate_grad_batches == 0:
                            self.optimizer[i].step()
                            self.optimizer[i].zero_grad()
                            if self.scheduler is not None:
                                self.scheduler[i].step(loss)
                        losses.append(loss.detach().mean())

                    if self.use_ema and self.global_step % (self.update_ema_interval*accumulate_grad_batches) == 0:
                        self.step_ema()

                    if len(self.optimizer) > 1:
                        pbar.set_description(
                            (
                                f'Epoch: [{epoch + 1} / {self.config.training.n_epochs}] '
                                f'iter: {self.global_step} loss-1: {losses[0]:.4f} loss-2: {losses[1]:.4f}'
                            )
                        )
                    else:
                        pbar.set_description(
                            (
                                f'Epoch: [{epoch + 1} / {self.config.training.n_epochs}] '
                                f'iter: {self.global_step} loss: {losses[0]:.4f}'
                            )
                        )


                end_time = time.time()
                elapsed_rounded = int(round((end_time-start_time)))
                self.logger("training time: " + str(datetime.timedelta(seconds=elapsed_rounded)))
                
                # validation
                if (epoch + 1) % self.config.training.validation_interval == 0 or (
                        epoch + 1) == self.config.training.n_epochs:
                    with torch.no_grad():
                        val_loader = create_val_dataloader(val_dataset=val_dataset,
                                                            batch_size=self.config.training.batch_size,
                                                            shuffle=False)
                        
                        average_loss = self.validation_epoch(val_loader, epoch)
                        

                # save checkpoint
                if (epoch + 1) % self.config.training.save_interval == 0 or \
                        (epoch + 1) == self.config.training.n_epochs:
                        with torch.no_grad():
                            
                            self.on_save_checkpoint(self.net, train_loader, val_loader, epoch, self.global_step)
                            model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='epoch_end')
                            
                            # save top_k checkpoints
                            model_ckpt_name = f'top_model_epoch_{epoch + 1}.pth'
                            optim_sche_ckpt_name = f'top_optim_sche_epoch_{epoch + 1}.pth'

                            if self.config.args.save_top and (epoch + 1) == self.config.training.n_epochs :
                                print("save top model start...")
                                # wandb.log('save top model start...')
                                top_key = 'top'
                                if top_key not in self.topk_checkpoints:
                                    print('top key not in topk_checkpoints')
                                    # wandb.log('top key not in topk_checkpoints')
                                    self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                    'model_ckpt_name': model_ckpt_name,
                                                                    'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                    print(f"saving top checkpoint: average_loss={average_loss} epoch={epoch + 1}")
                                    torch.save(model_states,
                                            os.path.join(self.config.result.ckpt_path, model_ckpt_name))
                                    torch.save(optimizer_scheduler_states,
                                            os.path.join(self.config.result.ckpt_path, optim_sche_ckpt_name))
                                else:
                                    if average_loss < self.topk_checkpoints[top_key]["loss"]:
                                        print("remove " + self.topk_checkpoints[top_key]["model_ckpt_name"])
                                        remove_file(os.path.join(self.config.result.ckpt_path,
                                                                self.topk_checkpoints[top_key]['model_ckpt_name']))
                                        remove_file(os.path.join(self.config.result.ckpt_path,
                                                                self.topk_checkpoints[top_key]['optim_sche_ckpt_name']))

                                        print(
                                        f"saving top checkpoint: average_loss={average_loss} epoch={epoch + 1}")

                                        self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                        'model_ckpt_name': model_ckpt_name,
                                                                        'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                        torch.save(model_states,
                                                os.path.join(self.config.result.ckpt_path, model_ckpt_name))
                                        torch.save(optimizer_scheduler_states,
                                                os.path.join(self.config.result.ckpt_path, optim_sche_ckpt_name))
                                if epoch + 1 == self.config.training.n_epochs:
                                    return os.path.join(self.config.result.ckpt_path, self.topk_checkpoints[top_key]['model_ckpt_name']), os.path.join(self.config.result.ckpt_path, self.topk_checkpoints[top_key]['optim_sche_ckpt_name'])
                                                
        except BaseException as e:
            print('str(Exception):\t', str(Exception))
            print('str(e):\t\t', str(e))
            print('repr(e):\t', repr(e))
            print('traceback.print_exc():')
            traceback.print_exc()
            print('traceback.format_exc():\n%s' % traceback.format_exc())

    @torch.no_grad()
    def test(self, task):
        
        low_candidates, low_scores = sampling_from_offline_data(x=self.offline_x,
                                                                y=self.offline_y,
                                                                n_candidates=self.config.testing.num_candidates, 
                                                                type=self.config.testing.type_sampling,
                                                                percentile_sampling=self.config.testing.percentile_sampling,
                                                                seed=self.config.args.seed)
        if self.use_ema:
            self.apply_ema()
        self.net.eval()
        
        task_to_min = {'TFBind8-Exact-v0': 0.0, 'TFBind10-Exact-v0': -1.8585268, 'AntMorphology-Exact-v0': -386.90036, 'DKittyMorphology-Exact-v0': -880.4585}
        task_to_max = {'TFBind8-Exact-v0': 1.0, 'TFBind10-Exact-v0': 2.1287067, 'AntMorphology-Exact-v0': 590.24445, 'DKittyMorphology-Exact-v0': 340.90985}
        task_to_best = {'TFBind8-Exact-v0': 0.43929616, 'TFBind10-Exact-v0': 0.005328223, 'AntMorphology-Exact-v0': 165.32648, 'DKittyMorphology-Exact-v0': 199.36252}
        
        oracle_y_min = task_to_min[self.config.task.name]
        oracle_y_max = task_to_max[self.config.task.name] 
        # normalize oracle_y_max by mean and std of offline data
        normalized_oracle_y_max = (oracle_y_max - self.mean_offline_y) / self.std_offline_y
        high_cond_scores = torch.full(low_scores.shape, normalized_oracle_y_max.item()*self.config.testing.alpha)
        
        high_candidates = self.sample(self.net, low_candidates, low_scores, high_cond_scores)

        # denormalize high_candidates
        high_candidates = high_candidates.cpu()
        denormalize_high_candidates = high_candidates * self.std_offline_x + self.mean_offline_x

        if task.is_discrete: 
            denormalize_high_candidates = denormalize_high_candidates.reshape(denormalize_high_candidates.shape[0],task.x.shape[1],task.x.shape[2])
        
        high_true_scores = task.predict(denormalize_high_candidates.numpy())
        # import pdb ; pdb.set_trace()
        final_score = (torch.from_numpy(high_true_scores) - oracle_y_min)/(oracle_y_max - oracle_y_min)    
        percentiles = torch.quantile(final_score, torch.tensor([1.0, 0.8, 0.5]), interpolation='higher') 
        
        return percentiles[0].item(), percentiles[1].item(), percentiles[2].item()