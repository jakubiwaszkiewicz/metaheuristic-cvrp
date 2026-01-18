import datetime
import os
from pathlib import Path
import pandas as pd
from decimal import Decimal, getcontext
from shapely.ops import unary_union
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from utils.constants import scale_factor, decimal_precision
from utils.data_repr import ChristmasTree

getcontext().prec = decimal_precision

def plot_trees_grid(trees_list, save_path, grid_size=(3, 3)):
    """Plot multiple tree arrangements in a single PNG as a grid."""
    n_plots = len(trees_list)
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for i, trees in enumerate(trees_list):
        ax = axes[i]
        colors = plt.cm.viridis([j / len(trees) for j in range(len(trees))])

        all_polygons = [tree.polygon for tree in trees]
        bounds = unary_union(all_polygons).bounds

        for index, tree in enumerate(trees):
            x_scaled, y_scaled = tree.polygon.exterior.xy
            x = [Decimal(val) / scale_factor for val in x_scaled]
            y = [Decimal(val) / scale_factor for val in y_scaled]
            ax.plot(x, y, color=colors[index])
            ax.fill(x, y, alpha=0.5, color=colors[index])
            ax.text(float(tree.center_x), float(tree.center_y), str(index), fontsize=6)

        minx = Decimal(bounds[0]) / scale_factor
        miny = Decimal(bounds[1]) / scale_factor
        maxx = Decimal(bounds[2]) / scale_factor
        maxy = Decimal(bounds[3]) / scale_factor

        width = maxx - minx
        height = maxy - miny
        side_length = max(width, height)
        square_x = minx if width >= height else minx - (side_length - width) / 2
        square_y = miny if height >= width else miny - (side_length - height) / 2
        bounding_square = Rectangle((float(square_x), float(square_y)), float(side_length), float(side_length), fill=False, edgecolor="red", linewidth=1, linestyle="--")
        ax.add_patch(bounding_square)

        padding = 0.1
        ax.set_xlim(float(square_x - Decimal(str(padding))), float(square_x + side_length + Decimal(str(padding))))
        ax.set_ylim(float(square_y - Decimal(str(padding))), float(square_y + side_length + Decimal(str(padding))))
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        score = side_length ** 2 / len(trees)
        ax.set_title(f"N={len(trees)} Score={score:.3f}", fontsize=8)

    for j in range(n_plots, rows*cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    submission_df = pd.read_csv("new_submission.csv")
    os.makedirs("plots", exist_ok=True)
    submission_df[['group_id', 'item_id']] = submission_df['id'].str.split('_', expand=True)
    grouped = submission_df.groupby('group_id')

    all_groups = [group for group, _ in grouped]
    all_groups.sort(key=lambda x: int(x))

    batch_size = 9
    save_time = datetime.datetime.now().strftime('%m-%d_%H:%M-%S')
    for batch_start in range(0, len(all_groups), batch_size):
        batch_groups = all_groups[batch_start:batch_start+batch_size]
        trees_list = []

        for group_id in batch_groups:
            group_data = submission_df[submission_df['group_id'] == group_id]
            trees = []
            for _, row in group_data.iterrows():
                x = row['x'].lstrip('s')
                y = row['y'].lstrip('s')
                deg = row['deg'].lstrip('s')
                trees.append(ChristmasTree(x, y, deg))
            trees_list.append(trees)
        dir_name = Path(f"plots/{save_time}")
        if not dir_name.exists():
            dir_name.mkdir(parents=True, exist_ok=True)
        save_path = dir_name / f"trees_{batch_start+1:03d}_{batch_start+len(batch_groups):03d}.png"
        print(f"Generating {save_path}")
        plot_trees_grid(trees_list, save_path, grid_size=(3, 3))

if __name__ == "__main__":
    main()
