import matplotlib.pyplot as plt
import numpy as np
import torch

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

def transform_explanations(E : np.ndarray, type: int = 1, scaler=None, unobserved=False, mode='train') -> np.ndarray:
    """
    Transform the explanations into a suitable format for training the anomaly detection model.
    This function should be implemented based on the specific requirements of the anomaly detection model.
    """
    eps = 1e-6  # Small constant to avoid division by zero

    if unobserved:
        # create a mask with True where the values are equal to -1, first column cannot be -1, so we can safely ignore it
        unobserved_mask = (E == -1)
        # substitute mask with safe probabilities (to avoid big intermediate calculations)
        E[unobserved_mask] = 0.5  # Replace -1 with a safe value
   
    if type == 1:
        # First transformation: Pop the first column (posterior probability of the target class)
        # Then divide each other column by this value (row-wise division) and compute the inverse (1/x) of the result
        posterior_probs = E[:, 0]
        transformed_E = posterior_probs[:, np.newaxis] / (E[:, 1:] + posterior_probs[:, np.newaxis] + eps)
    elif type == 2:
        # Second transformation: Each column after the first becomes the subtraction
        # of the first column and the original matrix with each column (row-wise subtraction)
        posterior_probs = E[:, 0]
        transformed_E = posterior_probs[:, np.newaxis] - E[:, 1:]
        transformed_E = np.concatenate((posterior_probs[:, np.newaxis], transformed_E), axis=1)
    elif type == 3:
        # Third transformation: similar to the second, but uses logit on each term
        posterior_probs = np.clip(E[:, 0], eps, 1 - eps)
        q_probs = np.clip(E[:, 1:], eps, 1 - eps)
        logit_p = np.log(posterior_probs / (1 - posterior_probs))
        logit_q = np.log(q_probs / (1 - q_probs))
        transformed_E = logit_p[:, np.newaxis] - logit_q
        transformed_E = np.concatenate((logit_p[:, np.newaxis], transformed_E), axis=1)
    elif type == 4:
        # Same as Type 1, but adding posterior probability of the target class as the first column
        posterior_probs = E[:, 0]
        transformed_E = posterior_probs[:, np.newaxis] / (E[:, 1:] + posterior_probs[:, np.newaxis] + eps)
        transformed_E = np.concatenate((posterior_probs[:, np.newaxis], transformed_E), axis=1)
    else:
        raise ValueError("Invalid transformation type")

    # If unobserved is True, replace the -1 values with the corresponding special values
    special_values = {1: 0, 2: -1, 3: 2*np.log((eps)/(1-eps)), 4: 0}
    if unobserved:
        if type > 1:
            transformed_E[unobserved_mask] = special_values[type]
        else:
            # exclude the first column from the mask for type 1
            transformed_E[unobserved_mask[:, 1:]] = special_values[type]

    if scaler is not None:
        if mode == 'train':
            transformed_E = scaler.fit_transform(transformed_E)
        elif mode == 'test':
            transformed_E = scaler.transform(transformed_E)
        else:
            raise ValueError("Invalid mode. Use 'train' or 'test'.")

    return transformed_E

def calc_anomaly_score(
    x: torch.Tensor, x_: torch.Tensor, score="mse", *,
    residual_mean: torch.Tensor = None, residual_cov: torch.Tensor = None,
):
    """Return MSE or ordinary (not squared) residual Mahalanobis distance.

    Matching inputs of shape (d,) return a zero-dimensional NumPy array;
    inputs of shape (n, d) return an array of shape (n,).
    For 'mah', supply the mean (d,) and unregularized covariance (d, d)
    fitted from reference absolute residuals, not from the samples being
    scored. Both statistics must share a floating dtype and the input device.
    Covariance must be symmetric positive definite; no regularization or
    pseudoinverse is applied.
    """
    if score not in ("mse", "mah"):
        raise ValueError("Invalid score type. Use 'mse' or 'mah'.")
    if not isinstance(x, torch.Tensor) or not isinstance(x_, torch.Tensor):
        raise TypeError("Inputs must be torch tensors.")
    if x.shape != x_.shape or x.ndim not in (1, 2) or x.shape[-1] == 0:
        raise ValueError("Inputs must have matching shapes (d,) or (n, d), with d > 0.")
    if x.device != x_.device or x.dtype != x_.dtype:
        raise ValueError("Inputs must share a device and dtype.")
    if not x.is_floating_point() or not torch.isfinite(x).all() or not torch.isfinite(x_).all():
        raise ValueError("Inputs must contain finite floating-point values.")

    if score == "mse":
        scores = (x - x_).square().mean(dim=-1)
    else:
        if residual_mean is None or residual_cov is None:
            raise ValueError("Mahalanobis scoring requires fitted residual_mean and residual_cov.")
        if not isinstance(residual_mean, torch.Tensor) or not isinstance(residual_cov, torch.Tensor):
            raise TypeError("Residual statistics must be torch tensors.")
        d = x.shape[-1]
        if residual_mean.shape != (d,) or residual_cov.shape != (d, d):
            raise ValueError("Residual mean and covariance must have shapes (d,) and (d, d).")
        if residual_mean.device != x.device or residual_cov.device != x.device:
            raise ValueError("Residual statistics must be on the input device.")
        if residual_mean.dtype != residual_cov.dtype or residual_cov.dtype not in (torch.float32, torch.float64):
            raise ValueError("Residual statistics must share a float32 or float64 dtype.")
        if not torch.isfinite(residual_mean).all() or not torch.isfinite(residual_cov).all():
            raise ValueError("Residual statistics must contain finite values.")
        if not torch.allclose(residual_cov, residual_cov.T):
            raise ValueError("Residual covariance must be symmetric.")
        try:
            chol = torch.linalg.cholesky(residual_cov)
        except torch.linalg.LinAlgError as exc:
            raise ValueError("Unregularized residual covariance must be positive definite.") from exc

        residuals = (x - x_).abs().to(dtype=residual_cov.dtype)
        centered = (residuals.unsqueeze(0) if x.ndim == 1 else residuals) - residual_mean
        whitened = torch.linalg.solve_triangular(chol, centered.T, upper=False)
        scores = whitened.square().sum(dim=0).sqrt()
        if x.ndim == 1:
            scores = scores[0]

    if not torch.isfinite(scores).all():
        raise ValueError("Anomaly scores are non-finite; check input magnitudes and covariance conditioning.")
    return scores.detach().cpu().numpy()
