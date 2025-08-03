import warnings

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')

import os
import json
import random
import tempfile
import sys
import glob
import torch
import time
import pickle
import shutil
import argparse
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image
from tqdm import tqdm
from torch.nn import L1Loss
from torchinfo import summary
from torch.cuda.amp import GradScaler, autocast
from monai.networks.nets import AutoencoderKL, VQVAE, PatchDiscriminator
from monai.losses import PatchAdversarialLoss, PerceptualLoss, GeneralizedDiceFocalLoss

from dosegen.data_processing import get_data_loaders
from dosegen.utils import load_config, create_2d_image_reconstruction_plot, create_gif_from_images, save_all_losses


class AutoEncoder:
    def __init__(self, config, latent_space_type='vae', print_summary=True):
        self.config = config
        self.print_summary = print_summary

        self.l1_loss = L1Loss()
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.seg_loss = GeneralizedDiceFocalLoss(softmax=True)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using device: CPU")

        if latent_space_type == 'vq':
            self.autoencoder = VQVAE(**config['vqvae_params']).to(self.device)
        elif latent_space_type == 'vae':
            self.autoencoder = AutoencoderKL(**config['vae_params']).to(self.device)
        else:
            raise ValueError("Invalid latent_space_type. Choose 'vq' or 'vae'.")

        if self.config['load_model_path']:
            # update loss_dict from previous training, as we are continuing training
            loss_pickle_path = os.path.join("/".join(self.config['load_model_path'].split('/')[:-2]), 'loss_dict.pkl')
            if os.path.exists(loss_pickle_path):
                with open(loss_pickle_path, 'rb') as file:
                    self.loss_dict = pickle.load(file)
        else:
            self.loss_dict = {'rec_loss': [], 'reg_loss': [], 'gen_loss': [], 'disc_loss': [], 'perc_loss': [],
                              'val_rec_loss': []}
            if 'label' in self.config['gen_mode']:
                self.loss_dict['seg_loss'] = []
                self.loss_dict['val_seg_loss'] = []

    @staticmethod
    def get_kl_loss(z_mu, z_sigma):
        kl_loss = 0.5 * (z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1)
        spatial_dims = list(range(1, len(z_mu.shape)))  # [1,2,3] for 2D, [1,2,3,4] for 3D
        kl_loss = torch.sum(kl_loss, dim=spatial_dims)
        return torch.sum(kl_loss) / kl_loss.shape[0]


    def train_one_epoch(self, epoch, train_loader, discriminator, perceptual_loss, optimizer_g, optimizer_d, scaler_g, scaler_d):
        self.autoencoder.train()
        discriminator.train()
        disable_prog_bar = self.config['output_mode'] == 'log' or not self.config['progress_bar']
        epoch_loss_dict = {'rec_loss': 0, 'reg_loss': 0, 'gen_loss': 0, 'disc_loss': 0, 'perc_loss': 0}
        if 'label' in self.config['gen_mode']:
            epoch_loss_dict['seg_loss'] = 0

        start = time.time()

        with tqdm(enumerate(train_loader), total=len(train_loader), ncols=150, disable=disable_prog_bar, file=sys.stdout) as progress_bar:
            progress_bar.set_description(f"Epoch {epoch}")

            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)
            for step, batch in progress_bar:
                images = batch["input"].to(self.device)
                step_loss_dict = {}

                reconstructions = self.train_generator_step(discriminator, epoch, images, optimizer_g, perceptual_loss,
                                                            scaler_g, step, step_loss_dict, train_loader)
                self.train_discriminator_step(discriminator, epoch, images, optimizer_d, reconstructions, scaler_d,
                                              step, step_loss_dict, train_loader)
                for key in step_loss_dict:
                    epoch_loss_dict[key] += step_loss_dict[key].item()

                progress_bar.set_postfix({key: value / (step + 1) for key, value in epoch_loss_dict.items()})

        epoch_loss_dict = {key: value / len(train_loader) for key, value in epoch_loss_dict.items()}

        if disable_prog_bar:
            end = time.time() - start
            print_string = f"Epoch {epoch} - Time: {time.strftime('%H:%M:%S', time.gmtime(end))}"
            for key in epoch_loss_dict:
                print_string +=  f" - {key}: {epoch_loss_dict[key]:.4f}"
            print(print_string)

        for key in epoch_loss_dict:
            self.loss_dict[key].append(epoch_loss_dict[key])

    def train_discriminator_step(self, discriminator, epoch, images, optimizer_d, reconstructions, scaler_d, step,
                                 step_loss_dict, train_loader):
        if epoch >= self.config['autoencoder_warm_up_epochs']:
            for param in discriminator.parameters():
                param.requires_grad = True
            for param in self.autoencoder.parameters():
                param.requires_grad = False

            with autocast(enabled=True):
                logits_fake = discriminator(reconstructions.contiguous().detach())[-1]
                loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                step_loss_dict['disc_loss'] = discriminator_loss * self.config['adv_weight']

            scaler_d.scale(step_loss_dict['disc_loss']).backward()

            if (step + 1) % self.config['grad_accumulate_step'] == 0 or (step + 1) == len(train_loader):
                # gradient clipping
                if self.config['grad_clip_max_norm']:
                    scaler_d.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(),
                                                   max_norm=self.config['grad_clip_max_norm'])
                scaler_d.step(optimizer_d)
                scaler_d.update()
                optimizer_d.zero_grad(set_to_none=True)

    def train_generator_step(self, discriminator, epoch, images, optimizer_g, perceptual_loss, scaler_g, step,
                             step_loss_dict, train_loader):
        for param in discriminator.parameters():
            param.requires_grad = False
        for param in self.autoencoder.parameters():
            param.requires_grad = True

        with autocast(enabled=True):
            if isinstance(self.autoencoder, VQVAE):
                reconstructions, quantization_loss = self.autoencoder(images)
                step_loss_dict['reg_loss'] = quantization_loss * self.config['q_weight']
            elif isinstance(self.autoencoder, AutoencoderKL):
                reconstructions, z_mu, z_sigma = self.autoencoder(images)
                step_loss_dict['reg_loss'] = self.get_kl_loss(z_mu, z_sigma) * self.config['kl_weight']

            step_loss_dict['perc_loss'] = perceptual_loss(reconstructions.float(), images.float()) * self.config['perc_weight']

            if not 'label' in self.config['gen_mode']:
                step_loss_dict['rec_loss'] = self.l1_loss(reconstructions.float(), images.float())
                loss_g = step_loss_dict['rec_loss'] + step_loss_dict['perc_loss'] + step_loss_dict['reg_loss']
            else:
                n_label_channels = self.config['dataset_config']['n_classes'] + 1
                not_seg_images = images[:, :-n_label_channels]
                seg_images = images[:, -n_label_channels:]
                not_seg_recons = reconstructions[:, :-n_label_channels]
                seg_recons = reconstructions[:, -n_label_channels:]

                step_loss_dict['rec_loss'] = self.l1_loss(not_seg_recons.float(), not_seg_images.float())
                step_loss_dict['seg_loss'] = self.seg_loss(seg_recons.float(), seg_images.float())

                loss_g = step_loss_dict['rec_loss'] + step_loss_dict['seg_loss'] + step_loss_dict['perc_loss'] + step_loss_dict['reg_loss']

            if epoch >= self.config['autoencoder_warm_up_epochs']:
                logits_fake = discriminator(reconstructions.contiguous().float())[-1]
                step_loss_dict['gen_loss'] = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False) * \
                                             self.config['adv_weight']
                loss_g += step_loss_dict['gen_loss']

        scaler_g.scale(loss_g).backward()
        if (step + 1) % self.config['grad_accumulate_step'] == 0 or (step + 1) == len(train_loader):
            # gradient clipping
            if self.config['grad_clip_max_norm']:
                scaler_g.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(),
                                               max_norm=self.config['grad_clip_max_norm'])
            scaler_g.step(optimizer_g)
            scaler_g.update()
            optimizer_g.zero_grad(set_to_none=True)
        return reconstructions

    def validate_one_epoch(self, val_loader, return_img_recon=False):
        self.autoencoder.eval()
        disable_prog_bar = self.config['output_mode'] == 'log' or not self.config['progress_bar']

        val_epoch_loss_dict = {'val_rec_loss': 0}
        if 'label' in self.config['gen_mode']:
            val_epoch_loss_dict['val_seg_loss'] = 0

        start = time.time()

        with tqdm(enumerate(val_loader), total=len(val_loader), ncols=100, disable=disable_prog_bar, file=sys.stdout) as val_progress_bar:
            for step, batch in val_progress_bar:
                images = batch["input"].to(self.device)

                with torch.no_grad():
                    with autocast(enabled=True):
                        reconstructions, *_ = self.autoencoder(images)

                        if not 'label' in self.config['gen_mode']:
                            recons_loss = self.l1_loss(reconstructions.float(), images.float())
                        else:
                            n_label_channels = self.config['dataset_config']['n_classes'] + 1
                            not_seg_images = images[:, :-n_label_channels]
                            seg_images = images[:, -n_label_channels:]
                            not_seg_recons = reconstructions[:, :-n_label_channels]
                            seg_recons = reconstructions[:, -n_label_channels:]

                            recons_loss = self.l1_loss(not_seg_recons.float(), not_seg_images.float())
                            seg_loss = self.seg_loss(seg_recons.float(), seg_images.float())

                val_epoch_loss_dict['val_rec_loss'] += recons_loss.item()
                if 'label' in self.config['gen_mode']:
                    val_epoch_loss_dict['val_seg_loss'] += seg_loss.item()

                val_progress_bar.set_postfix({key: value / (step + 1) for key, value in val_epoch_loss_dict.items()})

        val_epoch_loss_dict = {key: value / len(val_loader) for key, value in val_epoch_loss_dict.items()}

        if disable_prog_bar:
            end = time.time() - start
            print_string = f"Inference Time: {time.strftime('%H:%M:%S', time.gmtime(end))}"
            for key in val_epoch_loss_dict:
                print_string +=  f" - {key}: {val_epoch_loss_dict[key]:.4f}"
            print(print_string)

        for key in val_epoch_loss_dict:
            self.loss_dict[key].append(val_epoch_loss_dict[key])

        if return_img_recon:
            return images, reconstructions

    def get_optimizers_and_lr_schedules(self, discriminator):
        optimizer_g = torch.optim.Adam(params=self.autoencoder.parameters(), lr=self.config['ae_learning_rate'])
        optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=self.config['d_learning_rate'])

        if self.config["lr_scheduler"]:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.config["lr_scheduler"])  # Get the class dynamically
            g_lr_scheduler = scheduler_class(optimizer_g, **self.config["lr_scheduler_params"])
            d_lr_scheduler = scheduler_class(optimizer_d, **self.config["lr_scheduler_params"])
        else:
            g_lr_scheduler = None
            d_lr_scheduler = None

        return optimizer_g, optimizer_d, g_lr_scheduler, d_lr_scheduler

    def save_plots(self, image, reconstruction, plot_name):
        save_path = os.path.join(self.config['results_path'], 'plots', plot_name)
        os.makedirs(save_path, exist_ok=True)

        is_3d = image.ndim == 5  # (B, C, D, H, W) or (B, C, H, W)
        B, C = image.shape[:2]

        label_mode = 'label' in self.config['gen_mode']
        if label_mode:
            n_label_channels = self.config['dataset_config']['n_classes'] + 1
            image_labels = image[:, -n_label_channels:]
            recon_labels = reconstruction[:, -n_label_channels:]
            image = image[:, :-n_label_channels]
            reconstruction = reconstruction[:, :-n_label_channels]

            # Convert one-hot to mask
            image_masks = torch.argmax(image_labels, dim=1)  # (B, D, H, W) or (B, H, W)
            recon_masks = torch.argmax(recon_labels, dim=1)

        if is_3d:
            for idx in range(min(2, B)):
                for ch in range(C):
                    gif_images = []
                    img_vol = image[idx, ch]
                    recon_vol = reconstruction[idx, ch]
                    D = img_vol.shape[0]

                    for slice_idx in range(D):
                        fig, axs = plt.subplots(1, 2, figsize=(4, 2))
                        axs[0].imshow(img_vol[slice_idx].cpu(), cmap='gray', vmin=0, vmax=1)
                        axs[0].set_title("Image")
                        axs[0].axis("off")

                        axs[1].imshow(recon_vol[slice_idx].cpu(), cmap='gray', vmin=0, vmax=1)
                        axs[1].set_title("Reconstruction")
                        axs[1].axis("off")

                        buffer = BytesIO()
                        plt.tight_layout()
                        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
                        plt.close(fig)

                        buffer.seek(0)
                        gif_image = Image.open(buffer).copy()
                        gif_images.append(gif_image)
                        buffer.close()

                    gif_path = os.path.join(save_path, f"sample{idx}_channel{ch}.gif")
                    create_gif_from_images(gif_images, gif_path)

                if label_mode:
                    gif_images_label = []
                    img_mask_vol = image_masks[idx]
                    recon_mask_vol = recon_masks[idx]
                    D = img_mask_vol.shape[0]
                    for slice_idx in range(D):
                        fig, axs = plt.subplots(1, 2, figsize=(4, 2))
                        axs[0].imshow(img_mask_vol[slice_idx].cpu(), vmin=0, vmax=self.config['dataset_config']['n_classes'], cmap='hot')
                        axs[0].set_title("Image Label")
                        axs[0].axis("off")

                        axs[1].imshow(recon_mask_vol[slice_idx].cpu(), vmin=0, vmax=self.config['dataset_config']['n_classes'], cmap='hot')
                        axs[1].set_title("Recon Label")
                        axs[1].axis("off")

                        buffer = BytesIO()
                        plt.tight_layout()
                        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
                        plt.close(fig)

                        buffer.seek(0)
                        gif_image = Image.open(buffer).copy()
                        gif_images_label.append(gif_image)
                        buffer.close()

                    gif_path = os.path.join(save_path, f"sample{idx}_labels.gif")
                    create_gif_from_images(gif_images_label, gif_path)

        else:
            indices = random.sample(range(B), min(4, B))
            for idx in indices:
                fig, axs = plt.subplots(C, 2, figsize=(5, 2 * C))
                for ch in range(C):
                    axs[ch, 0].imshow(image[idx, ch].cpu(), cmap='gray', vmin=0, vmax=1)
                    axs[ch, 0].set_title(f"Image Ch {ch}")
                    axs[ch, 0].axis("off")

                    axs[ch, 1].imshow(reconstruction[idx, ch].cpu(), cmap='gray', vmin=0, vmax=1)
                    axs[ch, 1].set_title(f"Recon Ch {ch}")
                    axs[ch, 1].axis("off")

                fig.tight_layout()
                plt.savefig(os.path.join(save_path, f"sample{idx}.png"), dpi=300)
                plt.close(fig)

                if label_mode:
                    fig, axs = plt.subplots(1, 2, figsize=(5, 2))
                    axs[0].imshow(image_masks[idx].cpu(), vmin=0, vmax=self.config['dataset_config']['n_classes'], cmap='hot')
                    axs[0].set_title("Image Label")
                    axs[0].axis("off")

                    axs[1].imshow(recon_masks[idx].cpu(), vmin=0, vmax=self.config['dataset_config']['n_classes'], cmap='hot')
                    axs[1].set_title("Recon Label")
                    axs[1].axis("off")

                    fig.tight_layout()
                    plt.savefig(os.path.join(save_path, f"sample{idx}_labels.png"), dpi=300)
                    plt.close(fig)

    def save_model(self, epoch, validation_loss, optimizer, discriminator, disc_optimizer, scheduler=None,
                   disc_scheduler=None):
        save_path = os.path.join(self.config['results_path'], 'checkpoints')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        last_checkpoint_path = os.path.join(save_path, 'last_model.pth')
        checkpoint = {
            'epoch': epoch,
            'network_state_dict': self.autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_loss': validation_loss
        }

        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        checkpoint['discriminator_state_dict'] = discriminator.state_dict()
        checkpoint['disc_optimizer_state_dict'] = disc_optimizer.state_dict()
        if disc_scheduler:
            checkpoint['disc_scheduler_state_dict'] = disc_scheduler.state_dict()

        torch.save(checkpoint, last_checkpoint_path)

        best_checkpoint_path = os.path.join(save_path, 'best_model.pth')
        if os.path.isfile(best_checkpoint_path):
            best_checkpoint = torch.load(best_checkpoint_path)
            best_loss = best_checkpoint.get('validation_loss', float('inf'))
            if validation_loss < best_loss:
                torch.save(checkpoint, best_checkpoint_path)
        else:
            torch.save(checkpoint, best_checkpoint_path)

    def load_model(self, load_model_path, optimizer=None, scheduler=None, discriminator=None, disc_optimizer=None,
                   disc_scheduler=None, for_training=False):
        print(f'Loading model from {load_model_path}...')
        checkpoint = torch.load(load_model_path)
        self.autoencoder.load_state_dict(checkpoint['network_state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if discriminator and 'discriminator_state_dict' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        if disc_optimizer and 'disc_optimizer_state_dict' in checkpoint:
            disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])

        if disc_scheduler and 'disc_scheduler_state_dict' in checkpoint:
            disc_scheduler.load_state_dict(checkpoint['disc_scheduler_state_dict'])

        if for_training:
            return checkpoint['epoch'] + 1

    def train(self, train_loader, val_loader):
        scaler_g = GradScaler()
        scaler_d = GradScaler()
        total_start = time.time()
        start_epoch = 1
        plot_save_path = os.path.join(self.config['results_path'], 'plots')

        img_shape = self.config['ae_transformations']['patch_size']
        input_shape = (self.config['ae_batch_size'], self.autoencoder.encoder.in_channels, *img_shape)

        discriminator = PatchDiscriminator(**self.config['discriminator_params']).to(self.device)
        perceptual_loss = PerceptualLoss(**self.config['perceptual_params']).to(self.device)

        optimizer_g, optimizer_d, g_lr_scheduler, d_lr_scheduler = self.get_optimizers_and_lr_schedules(discriminator)

        if self.config['load_model_path']:
            start_epoch = self.load_model(self.config['load_model_path'], optimizer=optimizer_g, scheduler=g_lr_scheduler,
                                          discriminator=discriminator, disc_optimizer=optimizer_d, disc_scheduler=d_lr_scheduler,
                                          for_training=True)

        if self.print_summary:
            print("\nStarting training autoencoder model...")
            summary(self.autoencoder, input_shape, batch_dim=None, depth=3)
            summary(discriminator, input_shape, batch_dim=None, depth=3)
            summary(perceptual_loss, [input_shape, input_shape], batch_dim=None, depth=3)

        for epoch in range(start_epoch, self.config['n_epochs'] + 1):
            self.train_one_epoch(epoch, train_loader, discriminator, perceptual_loss, optimizer_g, optimizer_d, scaler_g, scaler_d)
            image, reconstruction = self.validate_one_epoch(val_loader, return_img_recon=True)
            save_all_losses(self.loss_dict, plot_save_path)
            save_loss_value = np.mean([l[-1] for l in self.loss_dict.values()])
            self.save_model(epoch, save_loss_value, optimizer_g, discriminator, optimizer_d,
                            scheduler=g_lr_scheduler, disc_scheduler=d_lr_scheduler)

            loss_pickle_path = os.path.join("/".join(plot_save_path.split('/')[:-1]), 'loss_dict.pkl')
            with open(loss_pickle_path, 'wb') as file:
                pickle.dump(self.loss_dict, file)

            if epoch % self.config['val_plot_interval'] == 0:
                self.save_plots(image, reconstruction, plot_name=f"epoch_{epoch}")

            if g_lr_scheduler:
                g_lr_scheduler.step()
                print(f"Adjusting learning rate of generator to {g_lr_scheduler.get_last_lr()[0]:.4e}.")

            if d_lr_scheduler:
                d_lr_scheduler.step()
                print(f"Adjusting learning rate of discriminator to {d_lr_scheduler.get_last_lr()[0]:.4e}.")

        total_time = time.time() - total_start
        print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")


def get_config_for_current_task(dataset_id, model_type, gen_mode, progress_bar, continue_training):
    preprocessed_dataset_path = glob.glob(os.getenv('dosegen_preprocessed') + f'/Task{dataset_id}*/')[0]

    config_path = os.path.join(preprocessed_dataset_path, 'dosegen_config.yaml')
    config = load_config(config_path)

    dataset_config_path = os.path.join(preprocessed_dataset_path, 'dataset.json')
    with open(dataset_config_path, 'r') as f:
        dataset_config = json.load(f)

    config = config[model_type]

    dataset_folder_name = preprocessed_dataset_path.split('/')[-2]
    results_path = os.path.join(os.getenv('dosegen_results'), dataset_folder_name, model_type, 'autoencoder', gen_mode)
    if os.path.exists(results_path) and not continue_training:
        raise FileExistsError(f"Results path {results_path} already exists.")

    last_model_path = os.path.join(results_path, 'checkpoints', 'last_model.pth')

    in_channels = 0
    if 'image' in gen_mode:
        in_channels += dataset_config['n_img_channels']
    if 'dose' in gen_mode:
        in_channels += 1
    if 'label' in gen_mode:
        in_channels += dataset_config['n_classes'] + 1

    config['vae_params']['in_channels'] = in_channels
    config['vae_params']['out_channels'] = in_channels
    config['discriminator_params']['in_channels'] = in_channels

    config['data_path'] = os.path.join(preprocessed_dataset_path, 'data')
    config['gen_mode'] = gen_mode
    config['dataset_config'] = dataset_config
    config['progress_bar'] = progress_bar
    config['output_mode'] = 'verbose'
    config['results_path'] = results_path
    config['load_model_path'] = last_model_path if continue_training else None

    return config


def validate_strict_combo(value):
    # Explicitly define allowed combinations
    allowed = {'image', 'dose', 'label', 'image-dose', 'image-label', 'dose-label', 'image-dose-label'}
    if value not in allowed:
        raise argparse.ArgumentTypeError(f"Invalid value '{value}'. Must be one of: {sorted(allowed)}")
    return value


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an Autoencoder Model to reconstruct images.")
    parser.add_argument("dataset_id", type=str, help="Dataset ID")
    parser.add_argument("splitting", choices=["train-val-test", "5-fold"],
                        help="Choose either 'train-val-test' for a standard split or '5-fold' for cross-validation.")
    parser.add_argument("model_type", choices=["2d", "3d"], help="Specify the model type: '2d' or '3d'.")
    parser.add_argument("gen_mode", type=validate_strict_combo, help="Str including items to generate, e.g., 'dose-label'.")
    parser.add_argument("-f", "--fold", type=int, choices=[0, 1, 2, 3, 4], required=False, default=None,
                        help="Specify the fold index (0-4) when using 5-fold cross-validation.")
    parser.add_argument("-l", "--latent_space_type", type=str, default="vae", choices=["vae", "vq"],
                        help="Type of latent space to use: 'vae' or 'vq'. Default is 'vae'.")
    parser.add_argument("-p", "--progress_bar", action="store_true", help="Enable progress bar (default: False)")
    parser.add_argument("-c", "--continue_training", action="store_true",
                        help="Continue training from the last checkpoint (default: False)")

    args = parser.parse_args()

    # Ensure --fold is provided only when --splitting is "5-fold"
    if args.splitting == "5-fold" and args.fold is None:
        parser.error("--fold is required when --splitting is set to '5-fold'")

    # Ensure --fold is None when --splitting is "train-val-test"
    if args.splitting == "train-val-test" and args.fold is not None:
        parser.error("--fold should not be provided when --splitting is set to 'train-val-test'")

    return args


def main():
    # Set temp dir BEFORE any other imports or logic
    temp_dir = tempfile.mkdtemp()  # Explicitly use local disk
    print(f"Using temp directory: {temp_dir}")
    os.environ["TMPDIR"] = temp_dir
    tempfile.tempdir = temp_dir

    try:
        args = parse_arguments()
        dataset_id = args.dataset_id
        splitting = args.splitting
        model_type = args.model_type
        gen_mode = args.gen_mode
        fold = args.fold
        latent_space_type = args.latent_space_type
        progress_bar = args.progress_bar
        continue_training = args.continue_training

        config = get_config_for_current_task(dataset_id, model_type, gen_mode, progress_bar, continue_training)

        transformations = config['ae_transformations']
        batch_size = config['ae_batch_size']
        train_loader, val_loader = get_data_loaders(config, dataset_id, splitting, batch_size, model_type, transformations, fold)

        model = AutoEncoder(config=config, latent_space_type=latent_space_type)
        model.train(train_loader=train_loader, val_loader=val_loader)

    finally:

        shutil.rmtree(temp_dir)
        print(f"Temp directory {temp_dir} removed.")

