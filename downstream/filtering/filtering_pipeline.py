import os
from collections import defaultdict
import argparse
import tqdm

# --- Configuration ---
EVENT_ROOT = "/home/bhzhang/Documents/visualize/event_former"
DIRECTIONS = ["forward"]


# --- Helper Functions ---
def read_scores_from_file(file_path):
    """Reads scores from a 'name,score' formatted text file into a dictionary."""
    scores = {}
    if not os.path.exists(file_path):
        print(f"Warning: Score file not found at {file_path}")
        return scores
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                name, score = parts
                try:
                    scores[name] = float(score)
                except ValueError:
                    print(f"Warning: Could not parse score for '{name}' in {file_path}")
    return scores

def get_event_folders(spatial_scores):
    """Groups event folders by their base name using keys from spatial_scores."""
    events_folder_map = defaultdict(list)
    for folder_name in spatial_scores.keys():
        try:
            event_name = folder_name.split("---")[0]
            events_folder_map[event_name].append(folder_name)
        except IndexError:
            print(f"Warning: Could not parse event name from folder '{folder_name}'")
    return events_folder_map


def select_and_sort_data(top_k_percent, final_n_samples, scales, spatial_scores, temporal_scores, event_folder_mapping):
    """Selects samples based on a two-stage temporal/spatial process and sorts them."""
    # Step 1: Filter temporal scores and separate by top_k
    print("Step 1: Filtering temporal scores...")
    sorted_temporal = sorted(temporal_scores.items(), key=lambda item: item[1], reverse=True)
    num_top_events = int(len(sorted_temporal) * (top_k_percent / 100.0))

    top_events = {event for event, score in sorted_temporal[:num_top_events]}
    bottom_events = {event for event, score in sorted_temporal[num_top_events:]}
    print(f"Split events into {len(top_events)} (top {top_k_percent}%) and {len(bottom_events)} (bottom).")

    # Step 2 & 3: Collect corresponding samples using the event_folder_mapping
    print("Step 2: Collecting samples using event folder mapping...")
    collected_samples = []
    all_events = top_events.union(bottom_events)

    for event_name in tqdm.tqdm(all_events, desc="Collecting samples"):
        if event_name not in event_folder_mapping:
            continue

        is_top = event_name in top_events
        folder_names = event_folder_mapping[event_name]

        for folder_name in folder_names:
            # Check if the folder has a score
            if folder_name not in spatial_scores:
                continue

            scale = folder_name.split('-scale-')[-1].split("---")[0]
            # scale = float(scale_str)


            # For top events, accept any scale in the provided list.
            # For bottom events, only accept the first scale in the list.
            if (is_top and scale in scales) or (not is_top and scale == scales[0]):
                collected_samples.append({
                    "folder_name": folder_name,
                    "spatial_score": spatial_scores[folder_name]
                })

    print(f"Collected {len(collected_samples)} total candidate samples.")

    # Step 4: Sort the collected samples by spatial score from high to low
    print("Step 3: Sorting samples by spatial score (high to low)...")
    collected_samples.sort(key=lambda x: x['spatial_score'], reverse=True)

    # Step 5: Select the top n_final samples
    print(f"Step 4: Selecting the top {final_n_samples} samples...")
    final_selection = collected_samples[:final_n_samples]
    print(f"Final selection contains {len(final_selection)} samples.")

    return final_selection


def save_selection_to_txt(selected_samples, output_file):
    """Writes the names of the selected samples to a text file."""
    print(f"Step 5: Writing {len(selected_samples)} names to '{output_file}'...")
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        for item in selected_samples:
            f.write(f"{item['folder_name']}\n")


def select_from_kw(data, kws):
    filtered_data = {}
    for k, v in data.items():
        if any(kw in k for kw in kws):
            filtered_data[k] = v
    return filtered_data


def main(args):
    """Main execution function."""
    print("Reading score files...")
    spatial_scores = read_scores_from_file(args.spatial_scores_file)
    spatial_scores = select_from_kw(spatial_scores, args.scales)
    spatial_scores = select_from_kw(spatial_scores, DIRECTIONS)
    temporal_scores = read_scores_from_file(args.temporal_scores_file)

    if not spatial_scores or not temporal_scores:
        print("Error: Could not read scores from one or both files. Exiting.")
        return

    print("Mapping event folders...")
    event_folder_mapping = get_event_folders(spatial_scores)
    if not event_folder_mapping:
        print("Error: No event folders found or EVENT_ROOT is incorrect. Exiting.")
        return

    selected_samples = select_and_sort_data(args.top_k, args.n_final, args.scales, spatial_scores, temporal_scores, event_folder_mapping)
    save_selection_to_txt(selected_samples, args.output_txt_file)

    print("\nProcessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select diverse event data based on pre-computed scores and save names to a file.")
    parser.add_argument('--top_k', type=float, default=50, help='Percentage of top samples to select based on temporal score (e.g., 50 for 50%).')
    parser.add_argument('--n_final', type=int, default=70000, help='Final number of candidates to select after sorting by spatial score.')
    parser.add_argument('--scales', nargs='+', type=float, default=["1.0", "2.0", "3.0"], help='List of scales to consider. All are used for top-tier, only the first is used for bottom-tier.')
    parser.add_argument('--spatial_scores_file', type=str, required=True, help='Path to the text file with spatial scores (e.g., pixel_dist.txt).')
    parser.add_argument('--temporal_scores_file', type=str, required=True, help='Path to the text file with temporal scores (e.g., scores.txt).')
    parser.add_argument('--output_txt_file', type=str, required=True, help='Path to the output text file for saving selected names.')
    args = parser.parse_args()

    main(args)