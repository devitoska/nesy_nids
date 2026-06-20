import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, 
                train_recon_losses, val_recon_losses, 
                train_kl_losses, val_kl_losses, dir, class_name):

    # Plot total loss (train and validation)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{dir}/loss_plot_{class_name}.png')
    plt.close()

    # Plot reconstruction loss (train and validation)
    plt.figure(figsize=(10, 5))
    plt.plot(train_recon_losses, label='Train Reconstruction Loss')
    plt.plot(val_recon_losses, label='Validation Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction Loss')
    plt.title('Training and Validation Reconstruction Loss')
    plt.legend()
    plt.savefig(f'{dir}/recon_loss_plot_{class_name}.png')
    plt.close()

    # Plot KL divergence loss (train and validation)
    plt.figure(figsize=(10, 5))
    plt.plot(train_kl_losses, label='Train KL Divergence Loss')
    plt.plot(val_kl_losses, label='Validation KL Divergence Loss')
    plt.xlabel('Epochs')
    plt.ylabel('KL Divergence Loss')
    plt.title('Training and Validation KL Divergence Loss')
    plt.legend()
    plt.savefig(f'{dir}/kl_loss_plot_{class_name}.png')
    plt.close()