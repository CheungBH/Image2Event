import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
from tqdm import tqdm

try:
    from downstream.distribution.hist_JS_optim import FlowDistributionScaler
except ImportError:
    from hist_JS_optim import FlowDistributionScaler


def main():
    parser = argparse.ArgumentParser(description="Visualize target flow scaling against source distribution")
    parser.add_argument('--source_flow_folder', type=str, required=True, help='Path to Source Domain flow folder (for global stats)')
    parser.add_argument('--target_flow_folder', type=str, required=True, help='Path to Target Domain flow folder (to visualize scaling)')
    parser.add_argument('--output_dir', type=str, default='scaling_check', help='Directory to save visualization results')
    parser.add_argument('--samples_per_image', type=int, default=200, help='Number of samples to take per source image for global distribution')
    parser.add_argument('--max_global_images', type=int, default=2000, help='Max source images to use for global stats')
    parser.add_argument('--max_target_images', type=int, default=20, help='Max target images to visualize')
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # List flow files
    source_files = sorted(glob.glob(os.path.join(args.source_flow_folder, "*.npy")))
    target_files = sorted(glob.glob(os.path.join(args.target_flow_folder, "*.npy")))
    
    if not source_files:
        print(f"❌ No .npy files found in source folder: {args.source_flow_folder}")
        return
    if not target_files:
        print(f"❌ No .npy files found in target folder: {args.target_flow_folder}")
        return

    print(f"Found {len(source_files)} source files and {len(target_files)} target files.")

    # --- 1. Compute Source Global Distribution ---
    print("\n🌍 Computing Source Global Distribution...")
    all_x = []
    all_y = []
    
    scaler = FlowDistributionScaler()
    
    # Subsample source images if too many
    if len(source_files) > args.max_global_images:
        print(f"Subsampling {args.max_global_images} source images...")
        source_files_sample = random.sample(source_files, args.max_global_images)
    else:
        source_files_sample = source_files
        
    for fpath in tqdm(source_files_sample, desc="Sampling Source"):
        flow = np.load(fpath).squeeze()
        # Ensure shape is (2, H, W)
        if flow.shape[2] == 2: # (H, W, 2)
            flow = flow.transpose(2, 0, 1)

        # Use stratified sampling
        xs, ys = scaler.stratified_sampling_from_flow(flow, n_samples=args.samples_per_image)
        all_x.extend(xs)
        all_y.extend(ys)

    source_x = np.array(all_x)
    source_y = np.array(all_y)
    print(f"Collected {len(source_x)} samples for source distribution.")

    # Determine range for plotting from Source
    x_min, x_max = np.percentile(source_x, [1, 99])
    y_min, y_max = np.percentile(source_y, [1, 99])
    
    # Add buffer
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_range = (x_min - x_span * 0.2, x_max + x_span * 0.2)
    y_range = (y_min - y_span * 0.2, y_max + y_span * 0.2)
    
    print(f"Source X Range: {x_range}")
    print(f"Source Y Range: {y_range}")
    
    # Pre-compute source histograms for efficiency
    src_hist_x, bin_edges_x = np.histogram(source_x, bins=100, range=x_range, density=True)
    bin_centers_x = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2
    
    src_hist_y, bin_edges_y = np.histogram(source_y, bins=100, range=y_range, density=True)
    bin_centers_y = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2

    # --- 2. Visualize Target Scaling ---
    print("\n🎲 Visualizing Target Scaling...")
    
    factors = [0.1, 0.25, 0.5, 0.75, 0.9, 1, 1.1, 2, 5, 8, 10]
    
    # Select target samples
    if len(target_files) > args.max_target_images:
        target_sample_files = random.sample(target_files, args.max_target_images)
    else:
        target_sample_files = target_files
        
    def plot_grid(component_name, target_data, src_hist, bin_centers, plot_range, filename, color):
        fig, axes = plt.subplots(4, 3, figsize=(20, 16))
        axes = axes.flatten()
        
        # Max density for y-axis scaling consistency (optional, but good)
        max_density = src_hist.max() * 1.5
        
        for i, ax in enumerate(axes):
            if i >= len(factors):
                ax.axis('off')
                continue
                
            factor = factors[i]
            scaled_target = target_data * factor
            
            # Plot Source (Reference)
            ax.fill_between(bin_centers, src_hist, alpha=0.3, color='gray', label='Source Global')
            
            # Plot Scaled Target
            ax.hist(scaled_target, bins=100, range=plot_range, density=True, 
                    alpha=0.6, color=color, label=f'Target * {factor}')
            
            ax.set_title(f"Scale: {factor}")
            ax.set_xlim(plot_range)
            # ax.set_ylim(0, max_density) # Optional: lock y-axis
            ax.set_yticks([])
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.2)
            
        plt.suptitle(f"Target: {filename} ({component_name}-Component)\nComparison with Source Global Distribution", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_name = os.path.basename(filename).replace('.npy', '')
        save_path = os.path.join(args.output_dir, f'{save_name}_{component_name}.png')
        plt.savefig(save_path, dpi=100)
        plt.close()

    for fpath in tqdm(target_sample_files, desc="Processing Targets"):
        try:
            flow = np.load(fpath)
            if flow.shape[2] == 2:
                flow = flow.transpose(2, 0, 1)
                
            flat_x = flow[0].flatten()
            flat_y = flow[1].flatten()
            
            # Downsample target pixels for plotting speed if needed (e.g. 50k)
            if len(flat_x) > 50000:
                indices = np.random.choice(len(flat_x), 50000, replace=False)
                flat_x = flat_x[indices]
                flat_y = flat_y[indices]
                
            # Plot X
            plot_grid('X', flat_x, src_hist_x, bin_centers_x, x_range, fpath, 'blue')
            
            # Plot Y
            plot_grid('Y', flat_y, src_hist_y, bin_centers_y, y_range, fpath, 'green')
            
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            
    print(f"\n✅ Done! Check results in {args.output_dir}")

if __name__ == "__main__":
    main()
