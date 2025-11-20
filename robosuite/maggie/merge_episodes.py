import zarr
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, '/home/maggie/research/object_centric_diffusion')
from utils.collect_utils import save_zarr

def main():
    input_dir = Path("/home/maggie/research/object_centric_diffusion/data/maggie_block/episodes")
    output_dir = Path("/home/maggie/research/object_centric_diffusion/data/train/maggie_block/all_variations/zarr")

    episode_folders = sorted([f for f in input_dir.iterdir() if f.name.startswith("episode_")],
                            key=lambda x: int(x.name.split("_")[1]))

    if not episode_folders:
        print("No episodes found!")
        return

    all_states = []
    all_states_in_world = []
    all_states_next = []
    all_states_next_in_world = []
    all_goals = []
    all_actions = []
    all_progress = []
    all_progress_binary = []
    all_task_stages = []
    all_variations = []
    episode_ends_cumulative = []

    cumulative_length = 0
    warnings = []

    print(f"Found {len(episode_folders)} episodes")

    for ep_folder in episode_folders:
        zarr_path = ep_folder / "zarr"
        if not zarr_path.exists():
            print(f"Warning: {ep_folder.name} has no zarr folder, skipping")
            continue

        root = zarr.open(str(zarr_path), 'r')

        state = np.array(root['data']['state'])
        state_in_world = np.array(root['data']['state_in_world'])
        state_next = np.array(root['data']['state_next'])
        state_next_in_world = np.array(root['data']['state_next_in_world'])
        goal = np.array(root['data']['goal'])
        action = np.array(root['data']['action'])
        progress = np.array(root['data']['progress'])
        progress_binary = np.array(root['data']['progress_binary'])
        task_stage = np.array(root['data']['task_stage'])
        variation = np.array(root['data']['variation'])

        episode_length = len(state)

        if episode_length < 3:
            warnings.append(f"{ep_folder.name} has only {episode_length} keyframes")

        all_states.extend(state)
        all_states_in_world.extend(state_in_world)
        all_states_next.extend(state_next)
        all_states_next_in_world.extend(state_next_in_world)
        all_goals.extend(goal)
        all_actions.extend(action)
        all_progress.extend(progress)
        all_progress_binary.extend(progress_binary)
        all_task_stages.extend(task_stage)
        all_variations.extend(variation)

        cumulative_length += episode_length
        episode_ends_cumulative.append(cumulative_length)

        print(f"  Loaded {ep_folder.name}: {episode_length} keyframes")

    total_keyframes = len(all_states)
    assert episode_ends_cumulative[-1] == total_keyframes, "Episode ends mismatch!"

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    save_zarr(
        str(output_dir),
        all_states,
        all_states_in_world,
        all_states_next,
        all_states_next_in_world,
        all_goals,
        all_actions,
        all_progress,
        all_progress_binary,
        all_task_stages,
        all_variations,
        episode_ends_cumulative,
    )

    print(f"\n{'='*50}")
    print(f"Merged {len(episode_folders)} episodes")
    print(f"Total keyframes: {total_keyframes}")
    if warnings:
        print(f"\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    print(f"\nOutput: {output_dir}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
