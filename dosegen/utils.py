import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os
import argparse
import yaml


def load_config(config_path):
    with open(config_path, "r") as file:
        config_file = yaml.safe_load(file)
        return config_file


def validate_strict_combo(value):
    # Explicitly define allowed combinations
    allowed = {'image', 'dose', 'label', 'image-dose', 'image-label', 'dose-label', 'image-dose-label'}
    if value not in allowed:
        raise argparse.ArgumentTypeError(f"Invalid value '{value}'. Must be one of: {sorted(allowed)}")
    return value


def create_2d_image_plot(image_slice, save_path):
    """
    Plots a single 2D slice of an image and its reconstruction side by side.
    If save_path is provided, the plot is saved; otherwise, the figure is returned as a BytesIO object.
    """
    plt.figure(figsize=(2, 2))
    plt.imshow(image_slice, cmap="gray")
    plt.title("Image")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Plot created successfully at {save_path}")


def create_2d_image_reconstruction_plot(image_slice, reconstruction_slice, save_path):
    """
    Plots a single 2D slice of an image and its reconstruction side by side.
    If save_path is provided, the plot is saved; otherwise, the figure is returned as a BytesIO object.
    """
    plt.figure(figsize=(4, 2))  # Side-by-side plots

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image_slice, vmin=0, vmax=1, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    # Plot reconstruction
    plt.subplot(1, 2, 2)
    plt.imshow(reconstruction_slice, vmin=0, vmax=1, cmap="gray")
    plt.title("Reconstruction")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Plot created successfully at {save_path}")


def create_gif_from_images(images, output_path, duration=200):
    """
    Creates a GIF from a list of PIL.Image objects.

    Args:
        images (list): List of PIL.Image objects.
        output_path (str): The path to save the output GIF.
        duration (int): The duration of each frame in milliseconds (default: 100ms).

    Returns:
        None
    """
    if not images:
        raise ValueError("No images provided to create GIF.")

    # Save the images as a GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

    print(f"GIF created successfully at {output_path}")


def save_main_losses(epoch_loss_list, val_epoch_loss_list, save_path):
    """
    Saves a plot of training and validation loss per epoch, handling cases where validation loss is logged at intervals.

    Args:
        epoch_loss_list (list): List of training loss values per epoch.
        val_epoch_loss_list (list): List of validation loss values collected at intervals.
        validation_interval (int): Interval at which validation losses are logged (e.g., every 20 epochs).
        save_path (str): Path to save the plot.
    """
    os.makedirs(save_path, exist_ok=True)
    save_plot_path = os.path.join(save_path, f"loss.png")

    epochs = range(1, len(epoch_loss_list) + 1)  # Epochs for training loss

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, epoch_loss_list, label="Training Loss", marker='o', linestyle='-')
    plt.plot(epochs, val_epoch_loss_list, label="Validation Loss", marker='o', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    # print(f"Loss plot saved at {save_path}")


def save_all_losses(loss_dict, save_path, log_scale=True):
    os.makedirs(save_path, exist_ok=True)
    save_plot_path = os.path.join(save_path, f"loss.png")

    epochs = range(1, len(loss_dict['rec_loss']) + 1)  # Epoch indices

    mapping_names_dict = {'rec_loss': 'Train Reconstruction Loss', 'val_rec_loss': 'Val Reconstruction Loss',
                          'reg_loss': 'Regularization Loss', 'gen_loss': 'Generator Loss',
                          'disc_loss': 'Discriminator Loss', 'perc_loss': 'Perceptual Loss'}

    if any(['seg' in key for key in loss_dict]):
        mapping_names_dict['seg_loss'] = 'Train Segmentation Loss'
        mapping_names_dict['val_seg_loss'] = 'Val Segmentation Loss'

    plt.figure(figsize=(10, 8))

    for key in mapping_names_dict:
        if key in loss_dict.keys():
            plt.plot(epochs, loss_dict[key], label=mapping_names_dict[key], linestyle='-')

    if log_scale:
        plt.yscale('log')
        plt.ylabel("log(loss)")
    else:
        plt.ylabel("loss")
    plt.xlabel("Epoch")
    plt.title("Losses per Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    # print(f"Loss plot saved at {save_path}")
