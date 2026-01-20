import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_temperature_grid(csv_list, save_path, grid_size=(3, 3)):
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for i, csv_path in enumerate(csv_list):
        ax = axes[i]

        df = pd.read_csv(csv_path, header=None, names=["temp", "value"])
        ax.plot(df["temp"], df["value"], color="blue")
        ax.set_title(Path(csv_path).stem, fontsize=8)
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Value")
        ax.grid(True)

    for j in range(len(csv_list), rows * cols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    input_dir = Path("all_scores")
    output_dir = Path("temp_plots")
    output_dir.mkdir(exist_ok=True)

    csv_files = sorted([str(p) for p in input_dir.glob("*.csv")])

    batch_size = 9
    save_time = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')

    for batch_start in range(0, len(csv_files), batch_size):
        batch = csv_files[batch_start:batch_start + batch_size]

        save_subdir = output_dir / save_time
        save_subdir.mkdir(parents=True, exist_ok=True)

        save_path = save_subdir / f"temps_{batch_start+1:03d}_{batch_start+len(batch):03d}.png"
        print(f"Generated {save_path}")

        plot_temperature_grid(batch, save_path, grid_size=(3, 3))


if __name__ == "__main__":
    main()
