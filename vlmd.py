import numpy as np
from sklearn import linear_model as lin_model

def vlmd(signal, num_modes, num_latents=None, alpha=100, reg_lambda=0.01, reg_rho=1,  sampling_rate=1, tolerance=1e-3, tau=0.9, max_iter=1000, verbose=False):
    # Variational Latent Mode Decomposition algorithm

    # Extract dimensions -> assumes that the signal is shaped channels x time-points
    """
    Variational Latent Mode Decomposition algorithm.

    Parameters
    ----------
    signal : ndarray (C x T)
        Input signal, where C is the number of channels and T is the number of time points.
    num_modes : int (K)
        Number of modes to extract.
    num_latents : int (L)
        Number of latent channels. If None (default), L = C
    alpha : float
        Regularization parameter > Frequency bandwith
    reg_lambda : float
        Regularization parameter > Recontsruction error
    reg_rho : float
        Regularization parameter > Sparsity regularization
    sampling_rate : float, optional
        Sampling rate of the signal.
    tolerance : float, optional
        Convergence tolerance.
    max_iter : int, optional
        Maximum number of iterations.
    tau : float, optional
        Step size for the dual variables.
    verbose : bool, optional
        Print running information.

    Returns
    -------
    modes_list : ndarray 
        The collection of the extracted modes. (K x L x T)
    latent_coefs : ndarray 
        Coefficient matrix. (L x C)
    omega_list : ndarray 
        Estimated mode center-frequencies. (iter x K)
    modes_hat : ndarray 
        Spectra of the modes. (K x C x F)
    """
    if signal is None or not isinstance(signal, np.ndarray) or signal.ndim != 2:
        raise ValueError("Signal must be a non-empty 2D numpy array (channels x t-points)")

    num_channels, num_tpoints = signal.shape

    if num_latents is None:
        num_latents = num_channels

    # Show dimensionality of the problem
    if verbose:
        print(
            f'___Parameters___ \n'
            f' Signal - Channels: {num_channels} Timepoints: {num_tpoints} \n'
            f' Model - Latent channels: {num_latents}\n'
            f' Model - Number of modes: {num_modes} ')

    # Initialize omegas
    omega_list = np.zeros((max_iter + 1, num_modes))

    # Latent signals
    latent_coefs = np.eye(num_latents, num_channels, dtype=np.float64) # (L x C)
    latent_sig = np.copy(signal[np.arange(num_latents) % num_channels, :])    # (L x T) Initialize from signal


    # --- Frequency domain ---
    num_fpoints = num_tpoints + 1
    f_points = np.linspace(0, 0.5, num_fpoints)

    signal_hat = _to_freq_domain(signal)
    latent_sig_hat = _to_freq_domain(latent_sig)

    modes_hat = np.zeros((num_modes, num_latents, num_fpoints), dtype=complex)

    # Set LASSO problem
    lasso = lin_model.Lasso(alpha=reg_lambda * 1e-3, warm_start=True, max_iter=1000, tol=0.2)
    lasso.coef_ = latent_coefs.T

    # === Latent Mode Decomposition ====
    # Start with empty dual variables
    gamma_hat = np.zeros((max_iter + 1, num_latents, num_fpoints), dtype=complex)
    residual_diff = tolerance + np.finfo(float).eps  # Stopping criterion
    n = 0  # Loop counter

    while n < max_iter and residual_diff > tolerance:
        # --- Latent Coefficients update ---
        lasso.fit(X=latent_sig.T, y=signal.T)
        latent_coefs = lasso.coef_.T

        # Normalization
        max_values = np.max(np.abs(latent_coefs), axis=1)
        for l, max_val in enumerate(max_values):
            if max_val > 1 :
                latent_coefs[l, :] = latent_coefs[l, :] / max_val

        # --- Latent signal update ---
        for l in range(num_latents):
            # Remove current contribution
            latent_sig_hat[l, :] = 0

            residual_hat = signal_hat - np.dot(latent_coefs.T, latent_sig_hat)

            latent_sig_hat[l, :] = (
                    (2.0 / reg_rho) * np.dot(latent_coefs[l, :], residual_hat) + np.sum(modes_hat[:, l, :], axis=0)
                    - gamma_hat[n, l, :]
            )
            latent_sig_hat[l, :] /= 1 + (2.0 / reg_rho) * np.sum(latent_coefs[l, :] ** 2)

        # Loop over the modes
        for k in range(num_modes):
            # --- Mode update ---
            modes_hat[k, :, :] = 0  # Remove contribution from the previous iteration

            # Update modes
            modes_hat[k, :, :] = latent_sig_hat - np.sum(modes_hat, axis=0) + gamma_hat[n, :, :]
            modes_hat[k, :, :] /= 1 + 4 * (alpha / reg_rho) * (f_points - omega_list[n, k]) ** 2

            # --- Update central frequencies ---
            module_mode_hat = np.abs(modes_hat[k, :, :]) ** 2

            if np.sum(module_mode_hat) < 1e-10:
                print('> No energy mode -> assuming constant')
                omega_list[n + 1, k] = 0

            else:
                omega_list[n + 1, k] = np.sum(np.dot(module_mode_hat, f_points))
                omega_list[n + 1, k] /= np.sum(module_mode_hat)

        # --- Update dual variables ---
        gamma_hat[n + 1, :, :] = gamma_hat[n, :, :] + tau  * (latent_sig_hat - np.sum(modes_hat, axis=0))

        # Update latent signal in the time domain
        latent_sig = _to_time_domain(latent_sig_hat)

        # Convergence check up
        residual_diff = np.sum(((omega_list[n + 1, :] - omega_list[n, :]) / (omega_list[n, :] + 1e-10)) ** 2)

        # Loop counter update
        n += 1

        # Print residual
        if verbose:
            print(f'Iteration {n:4.0f} - Residual: {residual_diff:.4e}')

    # Post-processing
    omega = omega_list[:n, :] / sampling_rate

    # Order the frequency list based on teh last results
    idx = np.argsort(omega[-1, :])
    omega = omega[:, idx]

    # Signal reconstruction
    modes_list = np.array([_to_time_domain(m_hat) for m_hat in modes_hat])

    # Order modes
    modes_list = modes_list[idx, :, :]

    return modes_list, latent_coefs, omega_list, modes_hat


def _to_freq_domain(signal, pad_mode='symmetric'):
    # Dimension
    tpoints = signal.shape[1]

    if pad_mode is None:
        full_signal = signal
    else:
        full_signal = np.pad(signal, pad_width=((0, 0), (tpoints // 2, tpoints - tpoints // 2)), mode=pad_mode)

    signal_hat = np.fft.fft(full_signal, axis=1)[:, :tpoints + 1]

    return signal_hat


def _to_time_domain(signal_hat, extended=False):
    channels, fpoints = signal_hat.shape
    red_ft = fpoints - 1

    # Construct Hermitian-symmetric assuming the signal is real
    full_signal_hat = np.zeros((channels, 2 * red_ft), dtype=complex)

    full_signal_hat[:, red_ft:] = signal_hat[:, :red_ft]
    full_signal_hat[:, :red_ft] = np.conj(signal_hat[:, red_ft:0:-1])

    # Inverse FFT to reconstruct the time-domain signal
    shifted_full_signal_hat = np.fft.ifftshift(full_signal_hat, axes=1)

    signal = np.real(np.fft.ifft(shifted_full_signal_hat, axis=1))

    if not extended:
        signal = signal[:, (red_ft // 2):(3 * red_ft // 2)]

    return signal