import matplotlib.pyplot as plt
import numpy as np

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

def transform_explanations(E : np.ndarray, type: int = 1, scaler=None) -> np.ndarray:
    """
    Transform the explanations into a suitable format for training the anomaly detection model.
    This function should be implemented based on the specific requirements of the anomaly detection model.
    """
    if type == 1:
        # First transformation: Pop the first column (posterior probability of the target class)
        # Then divide each other column by this value (row-wise division) and compute the inverse (1/x) of the result
        posterior_probs = E[:, 0]
        transformed_E = E[:, 1:] / posterior_probs[:, np.newaxis]
        transformed_E = 1 / transformed_E
    elif type == 2:
        # Second transformation: Each column after the first becomes the subtraction
        # of the first column from the original matrix with each column (row-wise subtraction)
        posterior_probs = E[:, 0]
        transformed_E = E[:, 1:] - posterior_probs[:, np.newaxis]
        transformed_E = np.concatenate((posterior_probs[:, np.newaxis], transformed_E), axis=1)
    elif type == 3:
        # Third transformation: similat to the second, but uses logit on each term
        posterior_probs = E[:, 0]
        logit_p = np.log(posterior_probs / (1 - posterior_probs))
        logit_q = np.log(E[:, 1:] / (1 - E[:, 1:]))
        transformed_E = logit_p[:, np.newaxis] - logit_q
        transformed_E = np.concatenate((logit_p[:, np.newaxis], transformed_E), axis=1)
    else:
        raise ValueError("Invalid transformation type")

    if scaler is not None:
        transformed_E = scaler.transform(transformed_E)

    return transformed_E