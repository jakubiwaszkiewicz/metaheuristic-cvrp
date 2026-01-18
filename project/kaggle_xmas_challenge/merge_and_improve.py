import os
import pandas as pd
from utils.data_repr import ChristmasTree
from main import (
    clone_trees,
    get_tree_list_side_length,
    get_total_score,
)

def parse_csv(csv_path) -> dict[str, list[ChristmasTree]]:
    result = pd.read_csv(csv_path)
    result['x'] = result['x'].str.strip('s')
    result['y'] = result['y'].str.strip('s')
    result['deg'] = result['deg'].str.strip('s')
    result[['group_id', 'item_id']] = result['id'].str.split('_', n=2, expand=True)

    dict_of_tree_list = {}
    dict_of_side_length = {}
    for group_id, group_data in result.groupby('group_id'):
        tree_list = [
            ChristmasTree(center_x=row['x'], center_y=row['y'], angle=row['deg'])
            for _, row in group_data.iterrows()
        ]
        dict_of_tree_list[group_id] = tree_list
        dict_of_side_length[group_id] = get_tree_list_side_length(tree_list)

    return dict_of_tree_list, dict_of_side_length
def find_submission_files():
    files = []
    for f in os.listdir("."):
        if f.lower().endswith(".csv") and "submission" in f.lower():
            files.append(f)
    return files


def merge_submissions(files):
    merged_tree_lists = {}
    merged_side_lengths = {}

    for file in files:
        print(f"Loading: {file}")
        dict_trees, dict_sides = parse_csv(file)

        for gid, trees in dict_trees.items():
            if gid not in merged_tree_lists:
                merged_tree_lists[gid] = trees
                merged_side_lengths[gid] = dict_sides[gid]
            else:
                # wybierz lepszy wariant
                old_side = merged_side_lengths[gid]
                new_side = dict_sides[gid]
                if new_side < old_side:
                    merged_tree_lists[gid] = trees
                    merged_side_lengths[gid] = new_side

    return merged_tree_lists, merged_side_lengths


def improve_solution(dict_of_tree_list, dict_of_side_length):
    current_score = get_total_score(dict_of_side_length)
    print(f"Initial score: {current_score}")

    for group_id_main in range(200, 1, -1):
        group_id_main = f"{group_id_main:03d}"
        print(f"Current box: {group_id_main}")

        group_id_prev = f"{int(group_id_main) - 1:03d}"
        best_side_length = dict_of_side_length[group_id_prev]
        best_tree_to_delete = None

        for tree_to_delete in range(int(group_id_main)):
            candidate_tree_list = clone_trees(dict_of_tree_list[group_id_main])
            del candidate_tree_list[tree_to_delete]

            candidate_side_length = get_tree_list_side_length(candidate_tree_list)

            if candidate_side_length < best_side_length:
                print(f" improvement {best_side_length:0.8f} -> {candidate_side_length:0.8f}")
                best_side_length = candidate_side_length
                best_tree_to_delete = tree_to_delete

        if best_tree_to_delete is not None:
            candidate_tree_list = clone_trees(dict_of_tree_list[group_id_main])
            del candidate_tree_list[best_tree_to_delete]

            dict_of_tree_list[group_id_prev] = candidate_tree_list
            dict_of_side_length[group_id_prev] = get_tree_list_side_length(candidate_tree_list)

    new_score = get_total_score(dict_of_side_length)
    print(f"Final score: {new_score} (improvement {current_score - new_score})")

    return dict_of_tree_list


def save_submission(dict_of_tree_list, filename="best_submission.csv"):
    rows = []
    for group_name, tree_list in dict_of_tree_list.items():
        for item_id, tree in enumerate(tree_list):
            rows.append({
                "id": f"{group_name}_{item_id}",
                "x": f"s{tree.center_x}",
                "y": f"s{tree.center_y}",
                "deg": f"s{tree.angle}",
            })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")


if __name__ == "__main__":
    files = find_submission_files()
    print("Found submission files:", files)

    dict_trees, dict_sides = merge_submissions(files)
    improved = improve_solution(dict_trees, dict_sides)
    save_submission(improved, "best_submission.csv")
