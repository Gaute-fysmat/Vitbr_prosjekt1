"""Script for running and plotting the FDM solution."""

from viz import create_animation, plot_snapshots, create_sensor_animation

from project import (
    load_config,
    solve_heat_equation, generate_training_data
    
)







def main():
    cfg = load_config("config.yaml")

    print("Solving heat equation with FDM...")
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)


    print("\nGenerating FDM visualizations...")
    plot_snapshots(
        x,
        y,
        t,
        T_fdm,
        save_path="output/fdm/fdm_snapshots.png",
    )
    create_animation(
        x, y, t, T_fdm, title="FDM", save_path="output/fdm/fdm_animation.gif"
    )

    create_sensor_animation( 
        sensor_data,
        title="Sensor measurements",
        cmap="inferno",
        save_path="output/sensor_animation.gif",
        fps=10,
    )







if __name__ == "__main__":
    main()
