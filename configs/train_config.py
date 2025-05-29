import torch


class GANTrainingConfig(object):
    """Configuration for GAN training and dataset"""
    # Model parameters
    num_classes: int = 2
    input_channels: int = 3
    output_channels: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Training parameters
    num_epochs: int = 1000
    batch_size: int = 16
    lr_generator: float = 1e-4
    lr_discriminator: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    
    # Loss weights
    adversarial_weight: float = 1.0
    segmentation_weight: float = 100.0
    adversarial_loss_type: str = "lsgan"  # lsgan, vanilla, wgan
    segmentation_loss_type: str = "combined"  # dice, ce, focal, combined
    
    # Learning rate scheduling
    lr_decay_step: int = 50
    lr_decay_gamma: float = 0.5
    
    # Logging and checkpointing
    log_interval: int = 100
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
    max_checkpoints: int = 1000
    
    # Early stopping
    early_stopping_patience: int = 20