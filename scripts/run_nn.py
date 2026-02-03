"""Script for training and plotting the NN model."""

import os

import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_nn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 4.4: Start
    #######################################################################
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)

    nn_params, a, = train_nn(sensor_data, cfg)

    T_nn = predict_grid(nn_params,x, y, t, cfg)

    
    create_animation(
    x, y, t, T_nn, title="FDM", save_path="output/fdm/NN_animation.gif"
    )


    #######################################################################
    # Oppgave 4.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
