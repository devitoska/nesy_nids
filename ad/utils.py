import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import genpareto

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
    
    if type is None:
        # do nothing, just return the original explanations
        transformed_E = E
    elif type == 1:
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
    
    if unobserved and type is not None:
        # If unobserved is True, replace the -1 values with the corresponding special values
        special_values = {1: 0, 2: -1, 3: 2*np.log((eps)/(1-eps)), 4: 0}
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

def select_youden_threshold(scores, correct, budget_percent=4):
    if budget_percent is None or not 0<=budget_percent<100:
        raise ValueError('Budget must be a percent from 0 to 99')
    scores=np.asarray(scores,dtype=float)
    correct=np.asarray(correct)
    nc=int(correct.sum())
    ni=len(correct)-nc
    order=np.argsort(scores,kind='stable')
    s=scores[order]
    ends=np.flatnonzero(np.r_[s[1:]!=s[:-1],True])
    ac=np.cumsum(correct[order])[ends];ai=ends+1-ac
    numerator=ac*ni-ai*nc
    feasible= (nc-ac)<=nc*budget_percent//100
    candidates=np.flatnonzero(feasible)
    chosen=candidates[np.flatnonzero(numerator[candidates]==numerator[candidates].max())[-1]]
    return float(s[ends[chosen]])

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
    Covariance must be symmetric. Before factorization, a diagonal ridge of
    1e-6 * max(mean diagonal variance, 1) is added without modifying the supplied
    covariance. The same regularization is used for calibration and inference.
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
        ridge = 1e-6 * residual_cov.diagonal().mean().clamp_min(1.0)
        regularized_cov = residual_cov + ridge * torch.eye(
            d, dtype=residual_cov.dtype, device=residual_cov.device
        )
        try:
            chol = torch.linalg.cholesky(regularized_cov)
        except torch.linalg.LinAlgError as exc:
            raise ValueError("Residual covariance is not positive definite after regularization.") from exc

        residuals = (x - x_).abs().to(dtype=residual_cov.dtype)
        centered = (residuals.unsqueeze(0) if x.ndim == 1 else residuals) - residual_mean
        whitened = torch.linalg.solve_triangular(chol, centered.T, upper=False)
        scores = whitened.square().sum(dim=0).sqrt()
        if x.ndim == 1:
            scores = scores[0]

    if not torch.isfinite(scores).all():
        raise ValueError("Anomaly scores are non-finite; check input magnitudes and covariance conditioning.")
    return scores.detach().cpu().numpy()


def EVT_cutoff(scores, threshold, EVT_rejection_rate, min_exceedances):

    eps = 1e-6  # Small constant
    
    with np.errstate(over="ignore", invalid="ignore"):
        # Compute the exceedances above the initial threshold
        exceedances = scores[scores > threshold] - threshold
        num_exceedances = exceedances.size

        if num_exceedances == 0:
            raise ValueError("EVT cannot be fitted: no scores exceed the initial threshold.")
        if not np.isfinite(exceedances).all():
            raise ValueError("EVT excesses are not finite; check score magnitudes.")
    
        tail_fraction = num_exceedances / scores.size

        if EVT_rejection_rate - tail_fraction > eps:
            try:
                raise ValueError(
                    f"EVT_rejection_rate ({EVT_rejection_rate:g}) exceeds the observed "
                    f"tail fraction ({tail_fraction:g}). Lower the target rate, select "
                    "a lower initial threshold, or use an empirical percentile instead."
                )
            except ValueError as exc:
                print("Defaulting to empirical threshold due to EVT target exceeding observed tail fraction.")
                return threshold, None
                
        if abs(EVT_rejection_rate - tail_fraction) <= eps:
            return threshold, None
        
        if num_exceedances < min_exceedances:
            raise ValueError(
                f"EVT requires at least {min_exceedances} exceedances; "
                f"only {num_exceedances} are available."
            )
        if np.all(exceedances == exceedances[0]):
            raise ValueError("EVT cannot be fitted to identical excesses.")
    
        try:
            shape, location, scale = genpareto.fit(exceedances, floc=0)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            raise ValueError("Failed to fit the generalized Pareto tail model.") from exc
        if (not np.isfinite([shape, location, scale]).all()
                or location != 0 or scale <= 0):
            raise ValueError("The fitted GPD has invalid shape, location, or scale parameters.")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            log_density = genpareto.logpdf(exceedances, c=shape, loc=0, scale=scale)
        if not np.isfinite(log_density).all():
            raise ValueError("The fitted GPD does not give finite density to all excesses.")
    
        # Convert the overall rejection target to a conditional tail probability.
        # isf handles both zero shape (the exponential limit) and near-zero shape.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            excess_cutoff = genpareto.isf(
                EVT_rejection_rate / tail_fraction, c=shape, loc=0, scale=scale,
            )
            final_threshold = threshold + excess_cutoff
        if not np.isfinite(final_threshold) or final_threshold <= threshold:
            raise ValueError("The fitted GPD did not produce a finite cutoff above the initial threshold.")
        
        return float(final_threshold), num_exceedances

def finalize_threshold(
    anomaly_scores, rejection_rate, EVT_rejection_rate=None, *, min_exceedances=2,
    correctness_mask=None):

    # validate anomaly_scores
    try:
        num_exceedances = None
        scores = np.asarray(anomaly_scores)
        if np.iscomplexobj(scores):
            raise ValueError("Complex scores are not supported.")
        scores = scores.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Anomaly scores must be a real numeric array.") from exc
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Anomaly scores must be a nonempty, finite, one-dimensional array.")

    # validate min_exceedances
    if (isinstance(min_exceedances, (bool, np.bool_))
                or not isinstance(min_exceedances, (int, np.integer))
                or min_exceedances < 2):
        raise ValueError("min_exceedances must be an integer of at least 2.")

    # validate rejection rate and EVT rejection rate
    auto_rejection = isinstance(rejection_rate, str) and rejection_rate == "auto"
    auto_evt = isinstance(EVT_rejection_rate, str) and EVT_rejection_rate == "auto"
    
    if auto_rejection and auto_evt:
        raise ValueError('rejection_rate and EVT_rejection_rate cannot both be "auto".')

    for name, rate in (("rejection_rate", rejection_rate),
                       ("EVT_rejection_rate", EVT_rejection_rate)):
        if name == "EVT_rejection_rate" and rate is None:
            continue
        if isinstance(rate, str) and rate == "auto":
            continue
        if (isinstance(rate, (bool, np.bool_))
                or not isinstance(rate, (int, float, np.integer, np.floating))
                or not np.isfinite(rate) or not 0 < rate < 1):
            raise ValueError(f"{name} must be a finite number strictly between 0 and 1.")

    # compute threshold with Youden's index if requested ("auto"), otherwise use empirical percentile
    correct = np.asarray(correctness_mask)
    if auto_rejection:
        n_correct = int(correct.sum())
        n_incorrect = scores.size - n_correct
        
        if correct.shape != scores.shape or correct.dtype.kind != "b":
            raise ValueError("Youden threshold requires a Boolean correctness_mask aligned with anomaly_scores.")
        if not n_correct:
            raise ValueError("Youden threshold requires at least one correct calibration sample.")
        # default to the 95th percentile of correct scores if there are no incorrect samples
        if not n_incorrect:
            threshold = float(np.percentile(scores[correct], 95))
        else:
            threshold = select_youden_threshold(scores, correct, budget_percent=4)
    else:
        threshold = float(np.percentile(scores, 100 * (1 - rejection_rate)))

    # validate threshold   
    if not np.isfinite(threshold):
        raise ValueError("The empirical threshold is not finite.")

    # compute EVT cutoff if requested, recalculating the threshold and number of exceedances
    if EVT_rejection_rate is not None:
        
        # default to Youden's index threshold if EVT rejection_rate is "auto"
        if EVT_rejection_rate == "auto":
            EVT_threshold = select_youden_threshold(scores, correct, budget_percent=4)
            # convert to an empirical percentile for EVT fitting
            EVT_rejection_rate = float(np.sum(scores > EVT_threshold)) / float(scores.size)
            # default to the 95th percentile of correct scores if there are no incorrect samples
            if EVT_rejection_rate == 0:
                EVT_rejection_rate = 0.05
        
        threshold, num_exceedances = EVT_cutoff(scores, threshold, EVT_rejection_rate, min_exceedances)
        
    return threshold, num_exceedances