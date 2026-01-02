import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
from tqdm import tqdm

# Add downstream to path to allow importing hist_JS_optim
sys.path.append(os.path.join(os.path.dirname(__file__), 'downstream'))

try:
    from hist_JS_optim import FlowDistributionScaler
except ImportError:
    print("Error: Could not import FlowDistributionScaler from downstream.hist_JS_optim")
    print("Please make sure you are running this script from the project root.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Verify if single optical flow distribution matches global distribution")
    parser.add_argument('--flow_folder', type=str, required=True, help='Path to the folder containing .npy flow files')
    parser.add_argument('--output_dir', type=str, default='distribution_check', help='Directory to save visualization results')
    parser.add_argument('--samples_per_image', type=int, default=200, help='Number of samples to take per image for global distribution')
    parser.add_argument('--max_global_images', type=int, default=2000, help='Max images to use for global stats')
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # List flow files
    flow_files = sorted(glob.glob(os.path.join(args.flow_folder, "*.npy")))
    if not flow_files:
        print(f"❌ No .npy files found in {args.flow_folder}")
        return

    print(f"Found {len(flow_files)} flow files.")

    # --- 1. Compute Global Distribution ---
    print("\n🌍 Computing Global Distribution...")
    all_x = []
    all_y = []
    
    scaler = FlowDistributionScaler()
    
    # Subsample images if too many
    global_files = flow_files
    if len(flow_files) > args.max_global_images:
        print(f"Subsampling {args.max_global_images} images for global stats...")
        global_files = random.sample(flow_files, args.max_global_images)
        
    for fpath in tqdm(global_files, desc="Sampling Global"):
        try:
            flow = np.load(fpath)
            # Ensure shape is (2, H, W)
            if flow.shape[2] == 2: # (H, W, 2)
                flow = flow.transpose(2, 0, 1)
            
            # Use stratified sampling from the scaler class
            xs, ys = scaler.stratified_sampling_from_flow(flow, n_samples=args.samples_per_image)
            all_x.extend(xs)
            all_y.extend(ys)
        except Exception as e:
            print(f"Warning: Error reading {fpath}: {e}")

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    print(f"Collected {len(all_x)} samples for global distribution.")

    # Determine range for plotting (exclude extreme outliers for better visualization)
    # Using 1st and 99th percentile
    x_min, x_max = np.percentile(all_x, [1, 99])
    y_min, y_max = np.percentile(all_y, [1, 99])
    
    # Add 10% buffer
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_range = (x_min - x_span * 0.1, x_max + x_span * 0.1)
    y_range = (y_min - y_span * 0.1, y_max + y_span * 0.1)
    
    print(f"X Range: {x_range}")
    print(f"Y Range: {y_range}")

    # --- 2. Select Single Samples ---
    print("\n🎲 Selecting Random Samples...")
    num_samples = 19
    if len(flow_files) < num_samples:
        print(f"Warning: Not enough files for {num_samples} samples, using {len(flow_files)}")
        sample_files = flow_files
    else:
        sample_files = random.sample(flow_files, num_samples)
        
    # Prepare plot data: list of (title, x_data, y_data)
    # First item is Global
    plot_data = [("Global Distribution", all_x, all_y)]
    
    for fpath in sample_files:
        name = os.path.basename(fpath)
        try:
            flow = np.load(fpath)
            if flow.shape[2] == 2:
                flow = flow.transpose(2, 0, 1)
            
            # For individual visualization, take more samples (e.g., 10k or all)
            flat_x = flow[0].flatten()
            flat_y = flow[1].flatten()
            
            # Downsample if too huge (e.g. > 20k) to save plotting time/memory
            if len(flat_x) > 20000:
                indices = np.random.choice(len(flat_x), 20000, replace=False)
                flat_x = flat_x[indices]
                flat_y = flat_y[indices]
                
            plot_data.append((name, flat_x, flat_y))
        except Exception as e:
            print(f"Error loading sample {name}: {e}")

    # --- 3. Visualization ---
    def plot_grid(component_name, data_idx, range_val, color):
        print(f"Plotting {component_name} distribution grid...")
        fig, axes = plt.subplots(4, 5, figsize=(24, 16))
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            if i >= len(plot_data):
                ax.axis('off')
                continue
                
            title, xs, ys = plot_data[i]
            data = xs if data_idx == 0 else ys
            
            # Plot histogram
            ax.hist(data, bins=60, range=range_val, density=True, alpha=0.7, color=color, edgecolor='none')
            
            # Styling
            if i == 0: # Global
                ax.set_title(f"★ {title} ★", fontsize=12, fontweight='bold', color='red')
                ax.set_facecolor('#f0f0f0')
            else:
                ax.set_title(title[-20:] if len(title)>20 else title, fontsize=9)
            
            ax.set_xlim(range_val)
            ax.set_yticks([]) # Hide y axis values for cleanliness
            ax.grid(True, alpha=0.3)
            
        plt.suptitle(f"Optical Flow {component_name}-Component Distribution Comparison\n(Global vs Single Samples)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = os.path.join(args.output_dir, f'distribution_grid_{component_name}.png')
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
        plt.close()

    # Plot X
    plot_grid('X', 0, x_range, 'blue')
    
    # Plot Y
    plot_grid('Y', 1, y_range, 'green')
    
    print("\n✅ Verification Done! Please check the output folder.")

if __name__ == "__main__":
    main()
