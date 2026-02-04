"""Training routines for NN and PINN models."""

import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm

from .config import Config
from .loss import bc_loss, data_loss, ic_loss, physics_loss
from .model import init_nn_params, init_pinn_params
from .optim import adam_step, init_adam
from .sampling import sample_bc, sample_ic, sample_interior
from tqdm import tqdm



def train_nn(
    sensor_data: jnp.ndarray, cfg: Config
) -> tuple[list[tuple[jnp.ndarray, jnp.ndarray]], dict]:
    """Train a standard neural network on sensor data only.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        params: Trained network parameters
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    nn_params = init_nn_params(cfg)
    adam_state = init_adam(nn_params)

    losses = {"total": [], "data": [], "ic": []}  # Fill with loss histories

    #######################################################################
    # Oppgave 4.3: Start

    def objekt_fn(nn_params):
            l_data = data_loss(nn_params, sensor_data, cfg)
            l_ic = ic_loss(nn_params, ic_epoch, cfg)
            total = cfg.lambda_data * l_data + cfg.lambda_ic * l_ic
            return total, (l_data, l_ic)
    objekt_fn = jax.jit(objekt_fn)
    
    # Din kode her
    for _ in tqdm(range(cfg.num_epochs), desc="Training NN"):
        ic_epoch, key = sample_ic(key, cfg)
        #objekt_ting= lambda nn_params: (cfg.lambda_data * data_loss(nn_params, sensor_data, cfg)
                        #+ cfg.lambda_ic * ic_loss(nn_params, ic_epoch, cfg))
        
        #  cfg.lambda_data, cfg.lambda_ic
        (total, (l_data, l_ic)), grads = jax.value_and_grad(objekt_fn, has_aux=True)(nn_params)

        nn_params, adam_state = adam_step(nn_params, grads, adam_state, lr=cfg.learning_rate)

        losses["total"].append(total)
        losses["data"].append(l_data)
        losses["ic"].append(l_ic)

    #######################################################################
    return nn_params, {k: jnp.array(v) for k, v in losses.items()}


    # Update the nn_params and losses dictionary

    #######################################################################
    # Oppgave 4.3: Slutt
    #######################################################################


def train_pinn(sensor_data: jnp.ndarray, cfg: Config) -> tuple[dict, dict]:
    """Train a physics-informed neural network.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration
    Returns:
        pinn_params: Trained parameters (nn weights + alpha)
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    pinn_params = init_pinn_params(cfg)
    opt_state = init_adam(pinn_params)

    losses = {"total": [], "data": [], "physics": [], "ic": [], "bc": []}

    #######################################################################
    # Oppgave 5.3: Start
    #######################################################################

    def PI_objekt_fn(pinn_params):
        l_data = data_loss(pinn_params["nn"], sensor_data, cfg)
        l_ic = ic_loss(pinn_params["nn"], ic_epoch, cfg)

        l_ph = physics_loss(pinn_params, interior_epoch, cfg)
        l_bc = bc_loss(pinn_params, bc_epoch, cfg)
        total = (cfg.lambda_data * l_data + cfg.lambda_ic * l_ic+ cfg.lambda_physics*l_ph + cfg.lambda_bc*l_bc)
        return total, (l_data, l_ic, l_ph, l_bc)
    PI_objekt_fn = jax.jit(PI_objekt_fn)


    for _ in tqdm(range(cfg.num_epochs), desc="Training PINN"):
        interior_epoch, key = sample_interior(key, cfg)
        ic_epoch, key = sample_ic(key, cfg)
        bc_epoch, key = sample_bc(key, cfg)

        (total, (l_data, l_ic, l_ph, l_bc)), grads = jax.value_and_grad(PI_objekt_fn, has_aux=True)(pinn_params)

        pinn_params, opt_state = adam_step(pinn_params, grads, opt_state, lr=cfg.learning_rate)

        losses["total"].append(total)
        losses["data"].append(l_data)
        losses["ic"].append(l_ic)
        losses["physics"].append(l_ph)
        losses["bc"].append(l_bc)


    # Update the nn_params and losses dictionary

    #######################################################################
    # Oppgave 5.3: Slutt
    #######################################################################

    return pinn_params, {k: jnp.array(v) for k, v in losses.items()}
