import os
import copy
import torch
from fastprogress import progress_bar
from utils import *
from modules import UNet_conditional
import logging
import wandb

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=10, c_in=3, c_out=3, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = UNet_conditional(c_in, c_out, num_classes=num_classes).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3):
        n = len(labels)
        logging.info(f"Sampling {n} new images....")
        model = self.ema_model if use_ema else self.model
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2

        return x

    # def train_step(self, loss):
    #     self.optimizer.zero_grad()
    #     self.scaler.scale(loss).backward()
    #     self.scaler.step(self.optimizer)
    #     self.scaler.update()
    #     self.ema.step_ema(self.ema_model, self.model)
    #     self.scheduler.step()

    # def one_epoch(self, train=True, use_wandb=False):
    #     avg_loss = 0.
    #     if train: self.model.train()
    #     else: self.model.eval()
    #     pbar = progress_bar(self.train_dataloader, leave=False)
    #     for i, (images, _, labels) in enumerate(pbar):
    #         with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
    #             images = images.to(self.device)
    #             labels = labels.to(self.device)
    #             t = self.sample_timesteps(images.shape[0]).to(self.device)
    #             x_t, noise = self.noise_images(images, t)
    #             if np.random.random() < 0.1:
    #                 labels = None
    #             predicted_noise = self.model(x_t, t, labels)
    #             loss = self.mse(noise, predicted_noise)
    #             avg_loss += loss
    #         if train:
    #             self.train_step(loss)
    #             if use_wandb: 
    #                 wandb.log({"train_mse": loss.item(),
    #                             "learning_rate": self.scheduler.get_last_lr()[0]})
    #         pbar.comment = f"MSE={loss.item():2.3f}"        
    #     return avg_loss.mean().item()

    def log_images(self, cond, iteration=0, use_wandb=False):
        "Log images to wandb and save them to disk"
        labels = cond.to(self.device)
        sampled_images = self.sample(use_ema=False, labels=labels)
        # ema_sampled_images = self.sample(use_ema=True, labels=labels)
        # plot_images(sampled_images)  #to display on jupyter if available
        # if use_wandb:
        #     wandb.log({"sampled_images":     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in sampled_images]})
            # wandb.log({"ema_sampled_images": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in ema_sampled_images]})

        # import ipdb; ipdb.set_trace()
        torch.save(sampled_images, f"/scratch/as3ek/spherical-diffusion/results/sample_{iteration}.pt")
        # torch.save(ema_sampled_images, f"/scratch/as3ek/spherical-diffusion/results/ema_sample_{iteration}.pt")

    def load(self, model_cpkt_path, ema_model_ckpt_path=""):
        self.model.load_state_dict(torch.load(model_cpkt_path, map_location='cpu'))
        if ema_model_ckpt_path != "":
            self.ema_model.load_state_dict(torch.load(ema_model_ckpt_path))

    def save_model(self, run_name, optimizer, iteration, use_wandb=False, epoch=-1):
        "Save model locally and on wandb"
        try:
            torch.save(self.model.state_dict(), os.path.join("outputs", run_name, f"ckpt_{iteration}.pt"))
        except:
            torch.save(self.model.module.state_dict(), os.path.join("outputs", run_name, f"ckpt_{iteration}.pt"))
        
        # try:
        #     torch.save(self.ema_model.state_dict(), os.path.join("outputs", run_name, f"ema_ckpt_{iteration}.pt"))
        # except:
        #     torch.save(self.ema_model.module.state_dict(), os.path.join("outputs", run_name, f"ema_ckpt_{iteration}.pt"))

        try:
            torch.save(optimizer.state_dict(), os.path.join("outputs", run_name, f"optimizer_ckpt_{iteration}.pt"))
        except:
            torch.save(optimizer.module.state_dict(), os.path.join("outputs", run_name, f"optimizer_ckpt_{iteration}.pt"))
        
        # if use_wandb:
        #     at = wandb.Artifact("model", type="model", description="Model weights for DDPM conditional", metadata={"epoch": epoch})
        #     at.add_dir(os.path.join("models", run_name))
        #     wandb.log_artifact(at)

    # def prepare(self, args):
    #     mk_folders(args.run_name)
    #     device = args.device
    #     self.train_dataloader = get_data(args)
    #     self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0.001)
    #     self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
    #                                              steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
    #     self.mse = nn.MSELoss()
    #     self.ema = EMA(0.995)
    #     self.scaler = torch.cuda.amp.GradScaler()

    # def fit(self, args):
    #     for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
    #         logging.info(f"Starting epoch {epoch}:")
    #         _  = self.one_epoch(train=True, use_wandb=args.use_wandb)
            
    #         # log predicitons
    #         if epoch % args.log_every_epoch == 0:
    #             self.log_images(use_wandb=args.use_wandb)
    #             # self.save_model(run_name=args.run_name, use_wandb=args.use_wandb, epoch=epoch)

    #     # save model
    #     self.save_model(run_name=args.run_name, use_wandb=args.use_wandb, epoch=epoch)