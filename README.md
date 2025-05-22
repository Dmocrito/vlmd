# Variational Latent Mode Decomposition (VLMD)

This code provides a python implementation of the VLMD algorithm. 

VLMD is a Python function for decomposing multivariate time series signals into sparse linear combinations of latent modes, according to the Latent Mode Decomposition (LMD) model.

The function takes in a multivariate time series signal and returns the decomposed latent modes, latent coefficients, and frequencies of the modes.
## Installation

To use VLMD, you will need to have Python 3.x and the following dependencies installed:

* `numpy`
* `scikit-learn`

You can install these dependencies using pip:
```bash
pip install numpy scikit-learn
```
## Usage

The `vlmd` function takes in a multivariate time series signal `signal` and returns the decomposed modes, frequencies, and latent coefficients.

```python
import numpy as np
from vlmd import vlmd

# Example signal (random)
signal = np.random.rand(10, 1000)  # 10 channels, 1000 time points

# Decompose the singal 
latent_modes, latent_coefs, omegas, latent_hat = vlmd(signal, num_modes=3, num_latents=5, alpha=100, reg_lambda=0.1, reg_rho=1)
```
## Parameters
* `signal`: ndarray
    - Multivariate time series signal (channels x samples)
* `num_modes`: int
    - Number of modes to extract
* `num_latents`: int
    - Number of latent channels
* `alpha`: float
    - Regularization parameter for frequency bandwidth 
* `reg_rho`: float
    - Regularization parameter for latent reconstruction error
* `reg_lambda`: float
    - Regularization parameter for sparsity (float)

**Optional parameters**
- `sampling_rate`: float
    - Sampling rate of the signal (default: 1)
- `tolerance`: float
    - Convergence tolerance (default: 1e-3)
- `max_iter`: int
    - Maximum number of iterations (default: 1000)
- `tau`: float
    - Step size for the dual variables (default: 0.9)
- `verbose`: bool
    - Print debug information (default: False)

## Returns

* `latent_modes`: ndarray
    - The collection of decomposed modes (K x C x T).
* `latent_coefs`: ndarray
    - Latent coefficients (L x C)
* `omegas`: ndarray 
    - Estimated mode center-frequencies per iteration (iter x K)
* `latent_hat`: ndarray
    - Spetcra of the latent modes (L x C x F)
## License

VLMD is released under the GNU General Public License (GPL) version 3. See the [LICENSE](LICENSE) file for details.
