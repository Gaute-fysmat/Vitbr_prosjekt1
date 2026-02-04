"""Script for training and plotting the PINN model."""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_pinn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 5.4: Start
    #######################################################################
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)

    pinn_params, a, = train_pinn(sensor_data, cfg)

    T_nn = predict_grid(pinn_params["nn"],x, y, t, cfg)


    create_animation(
    x, y, t, T_nn, title="FDM", save_path="output/fdm/piNN_animation_32n_5000epo.gif"
    )

    #######################################################################
    # Oppgave 5.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
