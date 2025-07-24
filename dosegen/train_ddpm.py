import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')

import os
import sys
import time
import glob
import traceback
import pickle
import tempfile
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from torch.cuda.amp import GradScaler, autocast
from itertools import combinations, cycle
from tqdm import tqdm
from io import BytesIO
from PIL import Image
from torchinfo import summary
from monai.utils import set_determinism
from generative.inferers import DiffusionInferer, LatentDiffusionInferer
from generative.networks.schedulers import DDPMScheduler
from generative.networks.nets import VQVAE
from generative.losses.perceptual import medicalnet_intensity_normalisation
from generative.metrics import FIDMetric, MMDMetric, SSIMMetric, MultiScaleSSIMMetric

from medimgen.data_processing import get_data_loaders
from medimgen.utils import load_config
from medimgen.autoencoderkl_with_strides import AutoencoderKL
from medimgen.diffusion_model_unet_with_strides import DiffusionModelUNet
from medimgen.utils import create_gif_from_images, save_all_losses


class LDM:
    def __init__(self, config, model_type, latent_space_type='vae', from2d=False):
        self.config = config
        self.model_type = model_type
        self.latent_space_type = latent_space_type
        self.from2d = from2d
        self.scale_factor = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using device: CPU")

        ae_model_type = self.model_type if not self.from2d else '2d'

        if self.from2d:
            print("Using 2D Autoencoder slices to train a 3D Latent Diffusion model.")

        print(f"Loading autoencoder checkpoint from {self.config['load_autoencoder_path']}...")
        if latent_space_type == 'vq':
            self.autoencoder = VQVAE(**self.config[ae_model_type]['vqvae_params']).to(self.device)
            checkpoint = torch.load(self.config['load_autoencoder_path'])
            self.autoencoder.load_state_dict(checkpoint['network_state_dict'])
            self.autoencoder.eval()
        elif self.latent_space_type == 'vae':
            self.autoencoder = AutoencoderKL(**self.config[ae_model_type]['vae_params']).to(self.device)
            checkpoint = torch.load(self.config['load_autoencoder_path'])
            self.autoencoder.load_state_dict(checkpoint['network_state_dict'])
            self.autoencoder.eval()
        else:
            raise ValueError("Invalid latent_space_type. Choose 'vq' or 'vae'.")
        print(f"Autoencoder epoch: {checkpoint['epoch']}")

        if latent_space_type == 'vq':
            self.codebook_min, self.codebook_max = self.get_codebook_min_max()

        self.ddpm = DiffusionModelUNet(**self.config[self.model_type]['ddpm_params']).to(self.device)

        # https://towardsdatascience.com/generating-medical-images-with-monai-e03310aa35e6
        self.scheduler = DDPMScheduler(**self.config[self.model_type]['time_scheduler_params'])

        if self.config['load_model_path']:
            # update loss_dict from previous training, as we are continuing training
            loss_pickle_path = os.path.join("/".join(self.config['load_model_path'].split('/')[:-2]), 'loss_dict.pkl')
            if os.path.exists(loss_pickle_path):
                with open(loss_pickle_path, 'rb') as file:
                    self.loss_dict = pickle.load(file)
        else:
            self.loss_dict = {'rec_loss': [], 'val_rec_loss': []}

    def get_codebook_min_max(self):
        codebook = self.autoencoder.quantizer.quantizer.embedding.weight.data  # [num_codes, embedding_dim]
        # Find min and max across all codebook vectors
        min_val = codebook.min().item()
        max_val = codebook.max().item()
        return min_val, max_val

    def codebook_min_max_normalize(self, tensor):
        return 2 * ((tensor - self.codebook_min) / (self.codebook_max - self.codebook_min)) - 1

    def codebook_min_max_renormalize(self, tensor):
        return ((tensor + 1) / 2) * (self.codebook_max - self.codebook_min) + self.codebook_min

    def get_inferer_and_latent_shape(self, train_loader):
        check_batch = next(iter(train_loader))['image']
        encoding_func = self.autoencoder.encode if self.latent_space_type == 'vq' else self.autoencoder.encode_stage_2_inputs

        if self.from2d:
            z_list = []
            for i in range(check_batch.shape[0]):
                sample_2d = check_batch[i].permute(1, 0, 2, 3).to(self.device)

                with torch.no_grad():
                    with autocast(enabled=True):
                        z_i = encoding_func(sample_2d)

                z_i = z_i.permute(1, 0, 2, 3)  # (C2, D, H2, W2)
                z_list.append(z_i)

            z = torch.stack(z_list, dim=0)  # (B, C2, D, H2, W2)
        else:
            with torch.no_grad():
                with autocast(enabled=True):
                    z = encoding_func(check_batch.to(self.device))

        if self.latent_space_type == 'vq':
            inferer = DiffusionInferer(self.scheduler)
        else:
            scale_factor = 1 / torch.std(z)
            self.scale_factor = scale_factor
            print(f"Scaling factor set to {scale_factor}")
            if self.from2d:
                inferer = DiffusionInferer(self.scheduler)
            else:
                inferer = LatentDiffusionInferer(self.scheduler, scale_factor=scale_factor)

        z_shape = tuple(z.shape)
        print(f"Latent shape: {z_shape}")
        return inferer, z_shape

    def get_optimizer_and_lr_schedule(self):
        optimizer = torch.optim.AdamW(params=self.ddpm.parameters(), lr=self.config[self.model_type]['ddpm_learning_rate'])
        # optimizer = torch.optim.SGD(self.ddpm.parameters(), self.config['ddpm_learning_rate'],
        #                               weight_decay=self.config['weight_decay'], momentum=0.99, nesterov=True)
        if self.config[self.model_type]["lr_scheduler"]:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.config[self.model_type]["lr_scheduler"])  # Get the class dynamically
            lr_scheduler = scheduler_class(optimizer, **self.config[self.model_type]["lr_scheduler_params"])
        else:
            lr_scheduler = None

        return optimizer, lr_scheduler

    def train_one_epoch(self, epoch, train_loader, optimizer, scaler):
        self.ddpm.train()
        self.autoencoder.eval()
        encoding_func = self.autoencoder.encode if self.latent_space_type == 'vq' else self.autoencoder.encode_stage_2_inputs
        scaling_func = self.codebook_min_max_normalize if self.latent_space_type == 'vq' else lambda x: x * self.scale_factor
        epoch_loss = 0
        disable_prog_bar = self.config['output_mode'] == 'log' or not self.config['progress_bar']
        start = time.time()

        with tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, disable=disable_prog_bar, file=sys.stdout) as progress_bar:
            progress_bar.set_description(f"Epoch {epoch}")

            optimizer.zero_grad(set_to_none=True)
            for step, batch in progress_bar:
                images = batch["image"].to(self.device)
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (images.shape[0],),
                                          device=images.device).long()

                with autocast(enabled=True):
                    with torch.no_grad():
                        if self.from2d:
                            z_list = []
                            for i in range(images.shape[0]):
                                volume = images[i]  # (C, D, H, W)
                                slices = volume.permute(1, 0, 2, 3)  # (D, C, H, W)
                                latents_i = encoding_func(slices).permute(1, 0, 2, 3)  # (C2, D, H2, W2)
                                z_list.append(latents_i)
                            latents = torch.stack(z_list, dim=0)  # (B, C2, D, H2, W2)
                        else:
                            latents = encoding_func(images)

                        latents_scaled = scaling_func(latents)

                    noise = torch.randn_like(latents_scaled).to(self.device)
                    noisy_latents = self.scheduler.add_noise(original_samples=latents_scaled, noise=noise, timesteps=timesteps)
                    noise_pred = self.ddpm(x=noisy_latents, timesteps=timesteps)

                    if self.scheduler.prediction_type == "v_prediction":
                        # Use v-prediction parameterization
                        target = self.scheduler.get_velocity(latents_scaled, noise, timesteps)
                    elif self.scheduler.prediction_type == "epsilon":
                        target = noise

                    loss = F.mse_loss(noise_pred.float(), target.float())

                scaler.scale(loss).backward()

                if (step + 1) % self.config[self.model_type]['grad_accumulate_step'] == 0 or (step +1) == len(train_loader):
                    # gradient clipping
                    if self.config[self.model_type]['grad_clip_max_norm']:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.ddpm.parameters(), max_norm=self.config[self.model_type]['grad_clip_max_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        # Log epoch loss
        if disable_prog_bar:
            end = time.time() - start
            print(f"Epoch {epoch} - Time: {time.strftime('%H:%M:%S', time.gmtime(end))} - "
                  f"Train Loss: {epoch_loss / len(train_loader):.4f}")

        self.loss_dict['rec_loss'].append(epoch_loss / len(train_loader))

    def validate_epoch(self, val_loader):
        self.ddpm.eval()
        self.autoencoder.eval()
        encoding_func = self.autoencoder.encode if self.latent_space_type == 'vq' else self.autoencoder.encode_stage_2_inputs
        scaling_func = self.codebook_min_max_normalize if self.latent_space_type == 'vq' else lambda x: x * self.scale_factor
        val_epoch_loss = 0
        disable_prog_bar = self.config['output_mode'] == 'log' or not self.config['progress_bar']
        start = time.time()

        with tqdm(enumerate(val_loader), total=len(val_loader), ncols=100, disable=disable_prog_bar, file=sys.stdout) as val_progress_bar:
            for step, batch in val_progress_bar:
                images = batch["image"].to(self.device)
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (images.shape[0],),
                                          device=images.device).long()

                with torch.no_grad():
                    with autocast(enabled=True):
                        if self.from2d:
                            z_list = []
                            for i in range(images.shape[0]):
                                volume = images[i]  # (C, D, H, W)
                                slices = volume.permute(1, 0, 2, 3)  # (D, C, H, W)
                                latents_i = encoding_func(slices).permute(1, 0, 2, 3)  # (C2, D, H2, W2)
                                z_list.append(latents_i)
                            latents = torch.stack(z_list, dim=0)  # (B, C2, D, H2, W2)
                        else:
                            latents = encoding_func(images)

                        latents_scaled = scaling_func(latents)

                        noise = torch.randn_like(latents_scaled).to(self.device)
                        noisy_latents = self.scheduler.add_noise(original_samples=latents_scaled, noise=noise,
                                                                 timesteps=timesteps)
                        noise_pred = self.ddpm(x=noisy_latents, timesteps=timesteps)

                        if self.scheduler.prediction_type == "v_prediction":
                            # Use v-prediction parameterization
                            target = self.scheduler.get_velocity(latents_scaled, noise, timesteps)
                        elif self.scheduler.prediction_type == "epsilon":
                            target = noise

                        val_loss = F.mse_loss(noise_pred.float(), target.float())

                val_epoch_loss += val_loss.item()
                val_progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})

        if disable_prog_bar:
            end = time.time() - start
            print(f"Time: {time.strftime('%H:%M:%S', time.gmtime(end))} - "
                  f"Validation Loss: {val_epoch_loss / len(val_loader):.4f}")

        self.loss_dict['val_rec_loss'].append(val_epoch_loss / len(val_loader))

    @staticmethod
    def get_perceptual_features(image, spatial_dims, perceptual_net):
        if spatial_dims == 2:
            # If input has just 1 channel, repeat channel to have 3 channels
            if image.shape[1]:
                image = image.repeat(1, 3, 1, 1)
            # Change order from 'RGB' to 'BGR'
            image = image[:, [2, 1, 0], ...]
            # Subtract mean used during training
            mean = [0.406, 0.456, 0.485]
            image[:, 0, :, :] -= mean[0]
            image[:, 1, :, :] -= mean[1]
            image[:, 2, :, :] -= mean[2]
            # Get model outputs
            with torch.no_grad():
                feature_image = perceptual_net.forward(image)
                # flattens the image spatially
                feature_image = feature_image.mean([2, 3], keepdim=False)
        else:
            image = medicalnet_intensity_normalisation(image)
            feature_image = perceptual_net.forward(image)
            feature_image = feature_image.mean([2, 3, 4], keepdim=False)

        return feature_image

    def validate_main(self, val_loader, z_shape, inferer, verbose, n_sampled_images, sampling_batch_size):
        spatial_dims = self.config[self.model_type]['ddpm_params']['spatial_dims']
        start = time.time()

        if spatial_dims == 2:
            perceptual_net = torch.hub.load("Warvito/radimagenet-models", model='radimagenet_resnet50', verbose=verbose).to(self.device)
        else:
            perceptual_net = torch.hub.load("Warvito/MedicalNet-models", model='medicalnet_resnet50_23datasets', verbose=verbose).to(self.device)
        perceptual_net.eval()

        ms_ssim = MultiScaleSSIMMetric(spatial_dims=spatial_dims, data_range=1.0, kernel_size=4)
        ssim = SSIMMetric(spatial_dims=spatial_dims, data_range=1.0, kernel_size=4)

        # Collect real features
        total_real_count = 0
        real_features = []
        val_iter = cycle(val_loader)
        while total_real_count < n_sampled_images:
            batch = next(val_iter)
            real_images = batch["image"].to(self.device)
            with torch.no_grad():
                real_eval_feats = self.get_perceptual_features(real_images, spatial_dims, perceptual_net)
            real_features.append(real_eval_feats)
            total_real_count += real_eval_feats.shape[0]
        real_features = torch.vstack(real_features)[:n_sampled_images]

        # Collect synthetic images and features
        total_synth_count = 0
        synth_features = []
        synth_images = []
        synth_z_shape = (sampling_batch_size,) + z_shape[1:]
        while total_synth_count < n_sampled_images:
            with torch.no_grad():
                with autocast(enabled=True):
                    temp_synth_images = self.sample_images(synth_z_shape, inferer, verbose, limited_samples=False)
                    synth_eval_feats = self.get_perceptual_features(temp_synth_images, spatial_dims, perceptual_net)
            synth_features.append(synth_eval_feats)
            synth_images.append(temp_synth_images)
            total_synth_count += synth_eval_feats.shape[0]
        synth_features = torch.vstack(synth_features)[:n_sampled_images]
        synth_images = torch.vstack(synth_images)[:n_sampled_images]

        # Compute FID
        fid = FIDMetric()
        fid_res = fid(synth_features, real_features)

        # Compute pairwise SSIM and MSSIM for synthetic images
        ms_ssim_scores = []
        ssim_scores = []
        idx_pairs = list(combinations(range(synth_images.shape[0]), 2))
        for idx_a, idx_b in idx_pairs:
            ms_ssim_val = ms_ssim(synth_images[[idx_a]], synth_images[[idx_b]])
            ssim_val = ssim(synth_images[[idx_a]], synth_images[[idx_b]])
            ms_ssim_scores.append(ms_ssim_val)
            ssim_scores.append(ssim_val)
        ms_ssim_scores = torch.cat(ms_ssim_scores, dim=0)
        ssim_scores = torch.cat(ssim_scores, dim=0)

        end = time.time() - start
        print(f"Time: {time.strftime('%H:%M:%S', time.gmtime(end))} - "
              f"FID: {fid_res.item():.4f} - "
              f"MS-SSIM: {ms_ssim_scores.mean():.4f} +- {ms_ssim_scores.std():.4f} - "
              f"SSIM: {ssim_scores.mean():.4f} +- {ssim_scores.std():.4f}")

        del perceptual_net, real_features, synth_images, synth_features

    def sample_images(self, z_shape, inferer, verbose=False, seed=None, limited_samples=True):
        self.ddpm.eval()
        self.autoencoder.eval()

        if limited_samples:
            # sample a maximum of 16 images for 2D, 2 for 3D
            max_n_samples = 16 if self.config[self.model_type]['ddpm_params']['spatial_dims'] == 2 else 2
            input_shape = [min(z_shape[0], max_n_samples), *z_shape[1:]]
        else:
            input_shape = z_shape

        if seed:
            # set seed for reproducible sampling
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                input_noise = torch.randn(*input_shape).to(self.device)
        else:
            input_noise = torch.randn(*input_shape).to(self.device)

        self.scheduler.set_timesteps(num_inference_steps=self.config[self.model_type]['time_scheduler_params']['num_train_timesteps'])

        with torch.no_grad():
            with autocast(enabled=True):
                if self.latent_space_type == 'vq':
                    generated_latents = inferer.sample(input_noise=input_noise, diffusion_model=self.ddpm,
                                                       scheduler=self.scheduler, verbose=verbose)
                    unscaled_latents = self.codebook_min_max_renormalize(generated_latents)
                    if self.from2d:
                        images_list = []
                        for i in range(unscaled_latents.shape[0]):
                            volume = unscaled_latents[i]  # (C2, D, H2, W2)
                            slices = volume.permute(1, 0, 2, 3)  # (D, C2, H2, W2)
                            quantized_latents, _ = self.autoencoder.quantize(slices)
                            images_i = self.autoencoder.decode(quantized_latents).permute(1, 0, 2, 3)  # (C, D, H, W)
                            images_list.append(images_i)
                        images = torch.stack(images_list, dim=0)  # (B, C, D, H, W)
                    else:
                        quantized_latents, _ = self.autoencoder.quantize(unscaled_latents)
                        images = self.autoencoder.decode(quantized_latents)
                elif self.latent_space_type == 'vae':
                    if self.from2d:
                        generated_latents = inferer.sample(input_noise=input_noise, diffusion_model=self.ddpm,
                                                           scheduler=self.scheduler, verbose=verbose)
                        unscaled_latents = generated_latents / self.scale_factor
                        images_list = []
                        for i in range(unscaled_latents.shape[0]):
                            volume = unscaled_latents[i]  # (C2, D, H2, W2)
                            slices = volume.permute(1, 0, 2, 3)  # (D, C2, H2, W2)
                            images_i = self.autoencoder.decode(slices).permute(1, 0, 2, 3)  # (C, D, H, W)
                            images_list.append(images_i)
                        images = torch.stack(images_list, dim=0)  # (B, C, D, H, W)
                    else:
                        images = inferer.sample(input_noise=input_noise, diffusion_model=self.ddpm,
                                                autoencoder_model=self.autoencoder, scheduler=self.scheduler,
                                                verbose=verbose)
                        # image = self.autoencoder.decode_stage_2_outputs(generated_latents)
        return images

    def save_plots(self, sampled_images, plot_name):
        save_path = os.path.join(self.config['results_path'], 'plots')
        os.makedirs(save_path, exist_ok=True)

        is_3d = len(sampled_images.shape) == 5

        if is_3d:
            # 3D case: assume shape (N, C, D, H, W)
            plot_save_path = os.path.join(save_path, f'{plot_name}.gif')
            num_volumes = min(sampled_images.shape[0], 2)
            num_slices = sampled_images.shape[2]
            gif_images = []

            # Loop over all slices in the depth dimension
            for slice_idx in range(num_slices):
                fig, axes = plt.subplots(1, num_volumes, figsize=(2 * num_volumes, 2))
                if num_volumes == 1:
                    axes = [axes]  # Ensure axes is iterable

                # For each volume (up to 2), plot the corresponding slice
                for vol_idx in range(num_volumes):
                    # Extract the slice for the given volume (assumes single channel)
                    slice_image = sampled_images.cpu()[vol_idx, 0, slice_idx, :, :]
                    axes[vol_idx].imshow(slice_image, vmin=0, vmax=1, cmap="gray")
                    axes[vol_idx].axis("off")

                plt.tight_layout(pad=0)
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()
                buffer.seek(0)
                gif_frame = Image.open(buffer).copy()
                gif_images.append(gif_frame)
                buffer.close()

            create_gif_from_images(gif_images, plot_save_path)

        else:
            # 2D case: assume shape (N, C, H, W)
            num_images = min(sampled_images.shape[0], 16)
            selected_images = sampled_images.cpu()[:num_images, 0, :, :]  # take the first channel
            columns = min(4, num_images)
            rows = (num_images + columns - 1) // columns

            fig, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2))
            # Ensure axes is a flat list for easy iteration
            if isinstance(axes, np.ndarray):
                axes = axes.flatten()
            else:
                axes = [axes]

            # Loop through grid positions
            for idx in range(rows * columns):
                ax = axes[idx]
                if idx < num_images:
                    ax.imshow(selected_images[idx], vmin=0, vmax=1, cmap="gray")
                    ax.axis("off")
                else:
                    ax.axis("off")  # Hide unused subplots

            plt.tight_layout(pad=0)
            plot_save_path = os.path.join(save_path, f'{plot_name}.png')
            plt.savefig(plot_save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Plot created successfully at {plot_save_path}")

    def save_model(self, epoch, validation_loss, optimizer, scheduler=None):
        save_path = os.path.join(self.config['results_path'], 'checkpoints')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        last_checkpoint_path = os.path.join(save_path, 'last_model.pth')
        checkpoint = {
            'epoch': epoch,
            'network_state_dict': self.ddpm.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_loss': validation_loss
        }

        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, last_checkpoint_path)

        best_checkpoint_path = os.path.join(save_path, 'best_model.pth')
        if os.path.isfile(best_checkpoint_path):
            best_checkpoint = torch.load(best_checkpoint_path)
            best_loss = best_checkpoint.get('validation_loss', float('inf'))
            if validation_loss < best_loss:
                torch.save(checkpoint, best_checkpoint_path)
        else:
            torch.save(checkpoint, best_checkpoint_path)

    def load_model(self, load_model_path, optimizer=None, lr_scheduler=None, for_training=False):
        print(f'Loading model from {load_model_path}...')
        checkpoint = torch.load(load_model_path)
        self.ddpm.load_state_dict(checkpoint['network_state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if lr_scheduler and 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if for_training:
            return checkpoint['epoch'] + 1

    def train(self, train_loader, val_loader):
        scaler = GradScaler()
        total_start = time.time()
        start_epoch = 1
        sample_seed = 42
        plot_save_path = os.path.join(self.config['results_path'], 'plots')
        sampling_batch_size = 50 if self.autoencoder.encoder.spatial_dims == 2 else 4
        n_sampled_images = 100 if self.autoencoder.encoder.spatial_dims == 2 else 40

        inferer, z_shape = self.get_inferer_and_latent_shape(train_loader)

        ae_img_shape = self.config[self.model_type]['ae_transformations']['patch_size']
        if self.from2d:
            ae_input_shape = (ae_img_shape[0], self.autoencoder.encoder.in_channels, *ae_img_shape[1:])
        else:
            ae_input_shape = (self.config[self.model_type]['ddpm_batch_size'], self.autoencoder.encoder.in_channels, *ae_img_shape)

        ddpm_input_shape = [(self.config[self.model_type]['ddpm_batch_size'], *z_shape[1:]), (self.config[self.model_type]['ddpm_batch_size'],)]

        optimizer, lr_scheduler = self.get_optimizer_and_lr_schedule()

        if self.config['load_model_path']:
            start_epoch = self.load_model(self.config['load_model_path'], optimizer=optimizer, lr_scheduler=lr_scheduler,
                                          for_training=True)

        print(f"\nStarting training ldm model...")
        summary(self.autoencoder, ae_input_shape, batch_dim=None, depth=3)
        summary(self.ddpm, ddpm_input_shape, batch_dim=None, depth=3)

        for epoch in range(start_epoch, self.config[self.model_type]['n_epochs'] + 1):
            self.train_one_epoch(epoch, train_loader, optimizer, scaler)
            self.validate_epoch(val_loader)
            save_all_losses(self.loss_dict, plot_save_path, log_scale=False)
            self.save_model(epoch, self.loss_dict['val_rec_loss'][-1], optimizer, lr_scheduler)

            loss_pickle_path = os.path.join("/".join(plot_save_path.split('/')[:-1]), 'loss_dict.pkl')
            with open(loss_pickle_path, 'wb') as file:
                pickle.dump(self.loss_dict, file)

            if epoch % self.config[self.model_type]['val_plot_interval'] == 0:
                sample_verbose = not (self.config['output_mode'] == 'log' or not self.config['progress_bar'])
                sampled_images = self.sample_images(z_shape, inferer, sample_verbose, seed=sample_seed)
                self.save_plots(sampled_images, plot_name=f"epoch_{epoch}")
                # for now validate only in 2D
                if self.config[self.model_type]['ddpm_params']['spatial_dims'] == 2:
                    self.validate_main(val_loader, z_shape, inferer, sample_verbose,
                                       n_sampled_images=n_sampled_images, sampling_batch_size=sampling_batch_size)

            if lr_scheduler:
                lr_scheduler.step()
                print(f"Adjusting learning rate to {lr_scheduler.get_last_lr()[0]:.4e}.")

        total_time = time.time() - total_start
        print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a Latent Diffusion model to generate images.")
    parser.add_argument("dataset_id", type=str, help="Dataset ID")
    parser.add_argument("splitting", choices=["train-val-test", "5-fold"],
                        help="Choose either 'train-val-test' for a standard split or '5-fold' for cross-validation.")
    parser.add_argument("model_type", choices=["2d", "3d"],
                        help="Specify the model type: '2d' or '3d'.")
    parser.add_argument("-f", "--fold", type=int, choices=[0, 1, 2, 3, 4, 5], required=False, default=None,
                        help="Specify the fold index (0-5) when using 5-fold cross-validation.")
    parser.add_argument("-l", "--latent_space_type", type=str, default="vae", choices=["vae", "vq"],
                        help="Type of latent space to use: 'vae' or 'vq'. Default is 'vae'.")
    parser.add_argument("-p", "--progress_bar", action="store_true", help="Enable progress bar (default: False)")
    parser.add_argument("-c", "--continue_training", action="store_true",
                        help="Continue training from the last checkpoint (default: False)")
    parser.add_argument("--from2d", action="store_true",
                        help="Train a 3D LDM with combined slices from 2D Autoencoder")
    parser.add_argument(
        "-l", "--labels_for",
        choices=["generation", "conditioning"],
        default="conditioning",
        help="Generate labels together with the doses or use them as condition to generate doses.",
        required=False
    )

    args = parser.parse_args()

    # Ensure --fold is provided only when --splitting is "5-fold"
    if args.splitting == "5-fold" and args.fold is None:
        parser.error("--fold is required when --splitting is set to '5-fold'")

    # Ensure --fold is None when --splitting is "train-val-test"
    if args.splitting == "train-val-test" and args.fold is not None:
        parser.error("--fold should not be provided when --splitting is set to 'train-val-test'")

    if args.from2d and args.model_type == "2d":
        parser.error("Argument --from2d can be used only if model_type = 3d")

    return args


def get_config_for_current_task(dataset_id, model_type, progress_bar, continue_training, from2d, initial_config=None):
    # preprocessed_dataset_path = glob.glob(os.getenv('nnUNet_preprocessed') + f'/Dataset{dataset_id}*/')[0]
    preprocessed_dataset_path = glob.glob(os.getenv('medimgen_preprocessed') + f'/Task{dataset_id}*/')[0]
    if not initial_config:
        config_path = os.path.join(preprocessed_dataset_path, 'medimgen_config.yaml')
        if os.path.exists(config_path):
            config = load_config(config_path)
        else:
            raise FileNotFoundError(
                f"There is no medimgen configuration file for Dataset {dataset_id}. First run: medimgen_plan")
    else:
        config = initial_config
    # config = config['2D'] if model_type == '2d' else config['3D']
    config['progress_bar'] = progress_bar
    config['output_mode'] = 'verbose'
    dataset_folder_name = preprocessed_dataset_path.split('/')[-2]

    main_results_folder = model_type if not from2d else 'hybrid'
    main_results_path = os.path.join(os.getenv('medimgen_results'), dataset_folder_name, main_results_folder)
    if from2d:
        trained_ae_path = os.path.join(os.getenv('medimgen_results'), dataset_folder_name, '2d', 'autoencoder', 'checkpoints', 'best_model.pth')
    else:
        trained_ae_path = os.path.join(main_results_path, 'autoencoder', 'checkpoints', 'best_model.pth')
    if not os.path.isfile(trained_ae_path):
        raise FileNotFoundError(f"No pretrained autoencoder found. You should first train an autoencoder in order to "
                                f"train a latent diffusion model")
    config['load_autoencoder_path'] = trained_ae_path

    results_path = os.path.join(main_results_path, 'ldm')
    if os.path.exists(results_path) and not continue_training:
        raise FileExistsError(f"Results path {results_path} already exists.")
    config['results_path'] = results_path
    last_model_path = os.path.join(results_path, 'checkpoints', 'last_model.pth')
    config['load_model_path'] = last_model_path if continue_training else None
    return config


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
        fold = args.fold
        latent_space_type = args.latent_space_type
        progress_bar = args.progress_bar
        continue_training = args.continue_training
        from2d = args.from2d

        config = get_config_for_current_task(dataset_id, model_type, progress_bar, continue_training, from2d)

        transformations = config[model_type]['ddpm_transformations']
        batch_size = config[model_type]['ddpm_batch_size']
        train_loader, val_loader = get_data_loaders(config[model_type], dataset_id, splitting, batch_size, model_type, transformations, fold)

        model = LDM(config=config, model_type=model_type, latent_space_type=latent_space_type, from2d=from2d)
        model.train(train_loader=train_loader, val_loader=val_loader)

    finally:

        shutil.rmtree(temp_dir)
        print(f"Temp directory {temp_dir} removed.")