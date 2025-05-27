import torch
from torch.utils.data import DataLoader, random_split
from data import CVC_CliniCDBDataset

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    print("Loading dataset...")
    dataset = CVC_CliniCDBDataset(config_path="data/configs/CVC_ClinicDB_config.py")
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4)
    print(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Model
    print("Loading model...")
    model = BaseGanModel.from_config()
    model.to(device)
    
    # Trainer
    print("Initializing trainer...")
    trainer = GANTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr_g=0.0001,
        lr_d=0.0004,
        num_epochs=100,
        n_critic=5,
        gp_weight=10.0,
        adv_weight=0.1,
        fm_weight=0.1,
        patience=20,
        output_dir="outputs",
        use_mixed_precision=True
    )
    print("Strarting training...")
    # Train
    trainer.train()


if __name__ == "__main__":
    main()