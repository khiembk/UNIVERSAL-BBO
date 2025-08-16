import torch.optim.lr_scheduler
from torch.utils.data import DataLoader

from Register import Registers
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer
from tqdm.autonotebook import tqdm


@Registers.runners.register_with_name('BBDMRunner')
class BBDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmnet = BrownianBridgeModel(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet

    def load_model_from_checkpoint(self):
        states = None
        if self.config.model.only_load_latent_mean_std:
            if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
                states = torch.load(self.config.model.model_load_path, map_location='cpu')
        else:
            states = super().load_model_from_checkpoint()

        if self.config.model.normalize_latent:
            if states is not None:
                self.net.ori_latent_mean = states['ori_latent_mean'].to(self.config.training.device[0])
                self.net.ori_latent_std = states['ori_latent_std'].to(self.config.training.device[0])
                self.net.cond_latent_mean = states['cond_latent_mean'].to(self.config.training.device[0])
                self.net.cond_latent_std = states['cond_latent_std'].to(self.config.training.device[0])
            else:
                if self.config.args.train:
                    self.get_latent_mean_std()

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num
        
    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                            mode='min',
                                                            verbose=True,
                                                            threshold_mode='rel',
                                                            **vars(config.model.BB.lr_scheduler)
)
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        if self.config.model.normalize_latent:
            model_states['ori_latent_mean'] = self.net.ori_latent_mean
            model_states['ori_latent_std'] = self.net.ori_latent_std
            model_states['cond_latent_mean'] = self.net.cond_latent_mean
            model_states['cond_latent_std'] = self.net.cond_latent_std
        return model_states, optimizer_scheduler_states

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        (x_high, y_high), (x_low, y_low) = batch
        
        torch.manual_seed(step)
        rand_mask = torch.rand(y_high.size())
        mask = (rand_mask <= self.config.training.classifier_free_guidance_prob)
        
        # mask y_high and y_low
        y_high[mask] = 0.
        y_low[mask] = 0.
            
        x_high = x_high.to(self.config.training.device[0])
        y_high = y_high.to(self.config.training.device[0])
        x_low = x_low.to(self.config.training.device[0])
        y_low = y_low.to(self.config.training.device[0])

        loss, additional_info = net(x_high, y_high, x_low, y_low)
        return loss

    @torch.no_grad()
    def sample(self, net, low_candidates, low_scores, high_cond_scores):
        low_candidates = low_candidates.to(self.config.training.device[0])
        low_scores = low_scores.to(self.config.training.device[0])
        high_cond_scores = high_cond_scores.to(self.config.training.device[0])
        high_candidates = net.sample(low_candidates, low_scores, high_cond_scores, clip_denoised=self.config.testing.clip_denoised, classifier_free_guidance_weight=self.config.testing.classifier_free_guidance_weight)
        
        return high_candidates