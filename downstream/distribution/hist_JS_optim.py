import numpy as np
import os
import glob
import json
import time
from tqdm import tqdm
from scipy.optimize import minimize_scalar
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt


class FlowDistributionScaler:
    """
    Optical flow distribution scaler: find optimal per-image scale factors by minimizing JS divergence.
    Optimize X and Y independently within [0.1, 10], starting from 1.0.
    """

    def __init__(self, source_folder=None, n_samples_per_image=1000, n_kde_components=1000):
        """
        Initialize scaler

        Parameters:
        -----------
        source_folder : str, optional
            Source-domain folder path. If provided, build source distribution immediately.
        n_samples_per_image : int
            Number of samples per source image
        n_kde_components : int
            Number of KDE components
        """
        self.n_samples_per_image = n_samples_per_image
        self.n_kde_components = n_kde_components
        self.source_dist_x = None
        self.source_dist_y = None
        self.grid_x = None
        self.grid_y = None
        self.dx = None
        self.dy = None
        self.source_stats = {}

        if source_folder:
            self.build_source_distribution(source_folder)

    def stratified_sampling_from_flow(self, flow, n_samples):
        """
        Stratified sampling from a single optical-flow map

        Parameters:
        -----------
        flow : numpy array, shape (2, H, W)
            Optical flow
        n_samples : int
            Number of samples

        Returns:
        -----------
        x_samples, y_samples : numpy arrays
            Samples of X and Y components
        """
        H, W = flow.shape[1], flow.shape[2]

        # 将图像分为4x4=16个区域
        n_regions_h, n_regions_w = 4, 4
        h_stride = H // n_regions_h
        w_stride = W // n_regions_w

        x_samples = []
        y_samples = []
        samples_per_region = max(1, n_samples // (n_regions_h * n_regions_w))

        for h_idx in range(n_regions_h):
            for w_idx in range(n_regions_w):
                # 区域边界
                h_start = h_idx * h_stride
                h_end = (h_idx + 1) * h_stride if h_idx < n_regions_h - 1 else H
                w_start = w_idx * w_stride
                w_end = (w_idx + 1) * w_stride if w_idx < n_regions_w - 1 else W

                # 从该区域随机抽样
                region_h = h_end - h_start
                region_w = w_end - w_start

                # 随机选择像素位置
                h_indices = np.random.randint(h_start, h_end, samples_per_region)
                w_indices = np.random.randint(w_start, w_end, samples_per_region)

                # 获取光流值
                for h, w in zip(h_indices, w_indices):
                    x_samples.append(flow[0, h, w])
                    y_samples.append(flow[1, h, w])

        return np.array(x_samples), np.array(y_samples)

    def build_kde_from_samples(self, samples, n_components=None):
        """
        Build KDE distribution from samples

        Parameters:
        -----------
        samples : numpy array
            Sampled data
        n_components : int, optional
            Number of KDE components; if None, use self.n_kde_components

        Returns:
        -----------
        kde_centers : numpy array
            KDE centers
        kde_weights : numpy array
            Kernel weights (uniform)
        bandwidth : float
            Kernel bandwidth
        """
        if n_components is None:
            n_components = self.n_kde_components

        # 从抽样数据中随机选择核中心
        if n_components < len(samples):
            indices = np.random.choice(len(samples), n_components, replace=False)
            kde_centers = samples[indices]
        else:
            kde_centers = samples
            n_components = len(samples)

        # 均匀权重
        kde_weights = np.ones(n_components) / n_components

        # 使用Silverman规则计算带宽
        sigma = np.std(samples)
        n = len(samples)
        bandwidth = 1.06 * sigma * (n ** (-0.2)) if sigma > 0 else 0.1

        return kde_centers, kde_weights, bandwidth

    def kde_density_on_grid(self, kde_centers, kde_weights, bandwidth, grid_points=1000):
        """
        Discretize KDE distribution onto a grid

        Parameters:
        -----------
        kde_centers : numpy array
            KDE centers
        kde_weights : numpy array
            Kernel weights
        bandwidth : float
            Kernel bandwidth
        grid_points : int
            Number of grid points

        Returns:
        -----------
        grid : numpy array
            Grid points
        prob_mass : numpy array
            Probability mass at each grid point
        """
        # Determine grid range (based on centers, slightly extended)
        x_min, x_max = np.min(kde_centers), np.max(kde_centers)
        range_extension = (x_max - x_min) * 0.1  # extend by 10%
        x_min -= range_extension
        x_max += range_extension

        # Create uniform grid
        grid = np.linspace(x_min, x_max, grid_points)
        dx = (x_max - x_min) / (grid_points - 1)

        # Compute KDE density on grid
        density = np.zeros_like(grid, dtype=float)
        sqrt_2pi = np.sqrt(2 * np.pi)

        for center, weight in zip(kde_centers, kde_weights):
            # Gaussian kernel contribution
            gaussian = np.exp(-0.5 * ((grid - center) / bandwidth) ** 2)
            gaussian /= (sqrt_2pi * bandwidth)
            density += weight * gaussian

        # Convert density to probability mass (multiply by dx) and normalize
        prob_mass = density * dx
        prob_mass /= np.sum(prob_mass)

        return grid, prob_mass, dx

    def build_source_distribution(self, source_folder, grid_points=1000):
        """
        Build source-domain distribution (offline)

        Parameters:
        -----------
        source_folder : str
            Source-domain folder path
        grid_points : int
            Number of grid points for discretization
        """
        print("=" * 70)
        print("Building source distribution")
        print("=" * 70)

        # 获取源域文件列表
        source_files = sorted(glob.glob(os.path.join(source_folder, "*.npy")))
        if not source_files:
            raise ValueError(f"No npy files found in source folder {source_folder}")

        print(f"Found {len(source_files)} source files")
        print(f"Sampling {self.n_samples_per_image} points per image")

        # 收集所有抽样点
        all_x_samples = []
        all_y_samples = []

        start_time = time.time()

        for i, file_path in enumerate(tqdm(source_files, desc="Processing source files")):
            # 加载光流图
            flow = np.load(file_path).squeeze()  # (2, H, W)

            # 分层抽样
            x_samples, y_samples = self.stratified_sampling_from_flow(
                flow, self.n_samples_per_image
            )

            all_x_samples.extend(x_samples)
            all_y_samples.extend(y_samples)

            # 每处理100个文件打印一次进度
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {i + 1}/{len(source_files)} files, elapsed {elapsed:.1f}s")

        all_x_samples = np.array(all_x_samples)
        all_y_samples = np.array(all_y_samples)

        print(f"X samples collected: {len(all_x_samples)} points")
        print(f"Y samples collected: {len(all_y_samples)} points")

        # 保存源域统计信息
        self.source_stats = {
            'x_mean': float(np.mean(all_x_samples)),
            'x_std': float(np.std(all_x_samples)),
            'x_median': float(np.median(all_x_samples)),
            'y_mean': float(np.mean(all_y_samples)),
            'y_std': float(np.std(all_y_samples)),
            'y_median': float(np.median(all_y_samples)),
            'n_samples_x': int(len(all_x_samples)),
            'n_samples_y': int(len(all_y_samples))
        }

        print("\nSource statistics:")
        print(f"  X: mean={self.source_stats['x_mean']:.4f}, "
              f"std={self.source_stats['x_std']:.4f}, "
              f"median={self.source_stats['x_median']:.4f}")
        print(f"  Y: mean={self.source_stats['y_mean']:.4f}, "
              f"std={self.source_stats['y_std']:.4f}, "
              f"median={self.source_stats['y_median']:.4f}")

        # 构建KDE分布
        print("\nBuilding KDE distributions...")
        kde_centers_x, kde_weights_x, bandwidth_x = self.build_kde_from_samples(all_x_samples)
        kde_centers_y, kde_weights_y, bandwidth_y = self.build_kde_from_samples(all_y_samples)

        print(f"KDE: X with {len(kde_centers_x)} kernels, bandwidth={bandwidth_x:.4f}")
        print(f"     Y with {len(kde_centers_y)} kernels, bandwidth={bandwidth_y:.4f}")

        # 离散化到网格
        print("\nDiscretizing KDE to grid...")
        self.grid_x, self.prob_mass_x, self.dx = self.kde_density_on_grid(
            kde_centers_x, kde_weights_x, bandwidth_x, grid_points
        )
        self.grid_y, self.prob_mass_y, self.dy = self.kde_density_on_grid(
            kde_centers_y, kde_weights_y, bandwidth_y, grid_points
        )

        print(f"Grid: X has {len(self.grid_x)} points, dx={self.dx:.6f}")
        print(f"      Y has {len(self.grid_y)} points, dy={self.dy:.6f}")

        elapsed = time.time() - start_time
        print(f"\nSource distribution built, total time {elapsed:.1f}s")
        print("=" * 70)

    def save_source_distribution(self, save_dir):
        """
        Save source distribution to files

        Parameters:
        -----------
        save_dir : str
            Output directory
        """
        os.makedirs(save_dir, exist_ok=True)

        # 保存分布数据
        np.savez_compressed(
            os.path.join(save_dir, 'source_distribution.npz'),
            grid_x=self.grid_x,
            prob_mass_x=self.prob_mass_x,
            grid_y=self.grid_y,
            prob_mass_y=self.prob_mass_y,
            dx=self.dx,
            dy=self.dy
        )

        # 保存统计信息
        with open(os.path.join(save_dir, 'source_stats.json'), 'w') as f:
            serializable_stats = {}
            for k, v in self.source_stats.items():
                if isinstance(v, (np.float32, np.float64)):
                    serializable_stats[k] = float(v)
                elif isinstance(v, np.integer):
                    serializable_stats[k] = int(v)
                elif isinstance(v, np.ndarray):
                    serializable_stats[k] = v.tolist()
                else:
                    serializable_stats[k] = v
            json.dump(serializable_stats, f, indent=2)

        print(f"Source distribution saved to: {save_dir}")

    def load_source_distribution(self, load_dir):
        """
        Load source distribution from files

        Parameters:
        -----------
        load_dir : str
            Directory to load from
        """
        data = np.load(os.path.join(load_dir, 'source_distribution.npz'))

        self.grid_x = data['grid_x']
        self.prob_mass_x = data['prob_mass_x']
        self.grid_y = data['grid_y']
        self.prob_mass_y = data['prob_mass_y']
        self.dx = data['dx']
        self.dy = data['dy']

        with open(os.path.join(load_dir, 'source_stats.json'), 'r') as f:
            self.source_stats = json.load(f)

        print(f"Source distribution loaded from {load_dir}")

    def compute_js_divergence(self, source_prob_mass, target_data, scale, grid, dx):
        """
        Compute JS divergence

        Parameters:
        -----------
        source_prob_mass : numpy array
            Probability mass of source distribution (discretized)
        target_data : numpy array
            Target data
        scale : float
            Scaling factor
        grid : numpy array
            Grid points
        dx : float
            Grid spacing

        Returns:
        -----------
        js : float
            JS divergence
        """
        # Scale target data
        scaled_target = target_data * scale

        # Compute histogram of target data on the grid
        target_hist, _ = np.histogram(
            scaled_target,
            bins=len(grid),
            range=(grid[0], grid[-1]),
            density=True
        )

        # Convert density to probability mass (multiply by dx) and normalize
        target_prob_mass = target_hist * dx
        target_prob_mass_sum = np.sum(target_prob_mass)

        if target_prob_mass_sum > 0:
            target_prob_mass /= target_prob_mass_sum
        else:
            # If all points fall outside the grid, return a large penalty
            return 1.0

        # Avoid zeros
        epsilon = 1e-10
        p = np.clip(source_prob_mass, epsilon, None)
        q = np.clip(target_prob_mass, epsilon, None)

        # Compute JS divergence
        m = 0.5 * (p + q)
        js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))

        return js

    def optimize_scale_for_single_image(self, target_flow, max_target_samples=10000):
        """
        Optimize scaling factors for a single target image

        Parameters:
        -----------
        target_flow : numpy array, shape (2, H, W)
            Target optical-flow map
        max_target_samples : int
            Max number of target samples (to accelerate optimization)

        Returns:
        -----------
        result : dict
            Optimization result containing scales, JS divergence, etc.
        """
        # 提取X和Y方向数据并展平
        target_x = target_flow[0].flatten()
        target_y = target_flow[1].flatten()

        # 如果数据量太大，进行随机抽样以加速优化
        if len(target_x) > max_target_samples:
            indices = np.random.choice(len(target_x), max_target_samples, replace=False)
            target_x = target_x[indices]
            target_y = target_y[indices]

        print(f"Target data: {len(target_x)} points")

        # 分别优化X和Y方向
        results = {'x': None, 'y': None}

        for direction, target_data, grid, prob_mass, dx in [
            ('x', target_x, self.grid_x, self.prob_mass_x, self.dx),
            ('y', target_y, self.grid_y, self.prob_mass_y, self.dy)
        ]:
            print(f"\nOptimizing {direction.upper()} direction...")

            # 定义目标函数
            def objective(scale):
                return self.compute_js_divergence(
                    prob_mass, target_data, scale, grid, dx
                )

            # 优化参数
            bounds = (0.1, 10.0)  # 搜索范围
            x0 = 1.0  # 初始值

            # 执行优化
            start_time = time.time()
            result = minimize_scalar(
                objective,
                bounds=bounds,
                method='bounded',
                options={'xatol': 1e-6, 'maxiter': 50, 'disp': False}
            )
            elapsed = time.time() - start_time

            if result.success:
                results[direction] = {
                    'optimal_scale': result.x,
                    'min_js': result.fun,
                    'iterations': result.nfev,
                    'time': elapsed
                }
                print(f"  Optimal scale: {result.x:.6f}")
                print(f"  Minimum JS divergence: {result.fun:.6f}")
                print(f"  Iterations: {result.nfev}, time: {elapsed:.3f}s")
            else:
                print(f"  Optimization failed for {direction.upper()}: {result.message}")
                # Use mean ratio as fallback
                source_mean = self.source_stats[f'{direction}_mean']
                target_mean = np.mean(target_data)
                fallback_scale = source_mean / target_mean if target_mean != 0 else 1.0
                fallback_scale = np.clip(fallback_scale, 0.1, 10.0)

                results[direction] = {
                    'optimal_scale': fallback_scale,
                    'min_js': objective(fallback_scale),
                    'iterations': 0,
                    'time': 0,
                    'fallback': True
                }
                print(f"  Using fallback scale: {fallback_scale:.6f}")

        # 计算总体结果
        if results['x'] and results['y']:
            overall_js = (results['x']['min_js'] + results['y']['min_js']) / 2

            # 计算原始分布的JS散度（缩放因子为1.0）
            original_js_x = self.compute_js_divergence(
                self.prob_mass_x, target_x, 1.0, self.grid_x, self.dx
            )
            original_js_y = self.compute_js_divergence(
                self.prob_mass_y, target_y, 1.0, self.grid_y, self.dy
            )
            original_js = (original_js_x + original_js_y) / 2

            # 计算改进比例
            if original_js > 0:
                improvement = (original_js - overall_js) / original_js * 100
            else:
                improvement = 0

            result_summary = {
                'scale_x': results['x']['optimal_scale'],
                'scale_y': results['y']['optimal_scale'],
                'js_x': results['x']['min_js'],
                'js_y': results['y']['min_js'],
                'js_overall': overall_js,
                'js_original': original_js,
                'improvement': improvement,
                'iterations_x': results['x']['iterations'],
                'iterations_y': results['y']['iterations'],
                'time_x': results['x']['time'],
                'time_y': results['y']['time'],
                'target_samples': len(target_x)
            }

            print("\n" + "=" * 50)
            print("Optimization summary:")
            print("=" * 50)
            print(f"Scale X: {result_summary['scale_x']:.6f}")
            print(f"Scale Y: {result_summary['scale_y']:.6f}")
            print(f"Overall JS: original={result_summary['js_original']:.6f}, "
                  f"optimized={result_summary['js_overall']:.6f}")
            print(f"Improvement: {result_summary['improvement']:.2f}%")
            print(f"Total optimization time: {result_summary['time_x'] + result_summary['time_y']:.3f}s")

            return result_summary
        else:
            raise ValueError("优化失败")

    def apply_scaling_to_flow(self, flow, scale_x, scale_y):
        """
        Apply scaling factors to an optical-flow map

        Parameters:
        -----------
        flow : numpy array, shape (2, H, W)
            Original optical flow
        scale_x, scale_y : float
            Scaling factors for X and Y

        Returns:
        -----------
        scaled_flow : numpy array
            Scaled optical flow
        """
        scaled_flow = flow.copy()
        scaled_flow[0] *= scale_x
        scaled_flow[1] *= scale_y
        return scaled_flow

    def process_target_folder(self, target_folder, output_folder=None,
                              save_scaled_flows=True, save_results=True):
        """
        Batch process all images in a target folder

        Parameters:
        -----------
        target_folder : str
            Target folder path
        output_folder : str, optional
            Output folder path
        save_scaled_flows : bool
            Whether to save scaled optical flows
        save_results : bool
            Whether to save optimization results

        Returns:
        -----------
        all_results : list
            Optimization results for all images
        """
        # 获取目标域文件列表
        target_files = sorted(glob.glob(os.path.join(target_folder, "*.npy")))
        if not target_files:
            raise ValueError(f"No npy files found in target folder {target_folder}")

        print("=" * 70)
        print(f"Processing target folder: {target_folder}")
        print(f"Found {len(target_files)} files")
        print("=" * 70)

        # 创建输出文件夹
        scaled_flows_dir = None
        if output_folder and (save_scaled_flows or save_results):
            os.makedirs(output_folder, exist_ok=True)
            if save_scaled_flows:
                scaled_flows_dir = os.path.join(output_folder, 'scaled_flows')
                os.makedirs(scaled_flows_dir, exist_ok=True)

        # 批量处理
        all_results = []
        total_start_time = time.time()

        for i, file_path in enumerate(tqdm(target_files, desc="Processing target files")):
            print(f"\nProcessing file {i + 1}/{len(target_files)}: {os.path.basename(file_path)}")

            # 加载目标光流图
            target_flow = np.load(file_path)

            # 优化缩放因子
            result = self.optimize_scale_for_single_image(target_flow)

            # 添加文件名信息
            result['filename'] = os.path.basename(file_path)
            result['filepath'] = file_path

            # 应用缩放（可选）
            if save_scaled_flows and scaled_flows_dir:
                scaled_flow = self.apply_scaling_to_flow(
                    target_flow, result['scale_x'], result['scale_y']
                )
                scaled_filename = f"scaled_{os.path.basename(file_path)}"
                np.save(os.path.join(scaled_flows_dir, scaled_filename), scaled_flow)
                result['scaled_file'] = os.path.join(scaled_flows_dir, scaled_filename)

            all_results.append(result)

        # 保存结果
        if save_results and output_folder:
            results_file = os.path.join(output_folder, 'optimization_results.json')
            with open(results_file, 'w') as f:
                # 转换numpy类型为Python类型
                serializable_results = []
                for result in all_results:
                    serializable_result = {}
                    for key, value in result.items():
                        if isinstance(value, (np.float32, np.float64)):
                            serializable_result[key] = float(value)
                        elif isinstance(value, np.ndarray):
                            serializable_result[key] = value.tolist()
                        elif isinstance(value, np.integer):
                            serializable_result[key] = int(value)
                        else:
                            serializable_result[key] = value
                    serializable_results.append(serializable_result)

                json.dump(serializable_results, f, indent=2)

            print(f"\nOptimization results saved to: {results_file}")

        # 分析统计结果
        self._analyze_batch_results(all_results)

        total_time = time.time() - total_start_time
        print(f"\nBatch processing completed!")
        print(f"Total files: {len(target_files)}")
        print(f"Successfully processed: {len(all_results)}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average per frame: {total_time / len(all_results):.3f}s")
        print("=" * 70)

        return all_results

    def _analyze_batch_results(self, results):
        """Analyze batch results"""
        if not results:
            print("No results to analyze")
            return

        scales_x = np.array([r['scale_x'] for r in results])
        scales_y = np.array([r['scale_y'] for r in results])
        improvements = np.array([r['improvement'] for r in results])
        js_overall = np.array([r['js_overall'] for r in results])
        js_original = np.array([r['js_original'] for r in results])

        print("\n" + "=" * 70)
        print("Batch statistics")
        print("=" * 70)

        print(f"Scaling factor X:")
        print(f"  Mean: {np.mean(scales_x):.4f}, Std: {np.std(scales_x):.4f}")
        print(f"  Median: {np.median(scales_x):.4f}")
        print(f"  Range: [{np.min(scales_x):.4f}, {np.max(scales_x):.4f}]")
        print(f"  95% CI: [{np.percentile(scales_x, 2.5):.4f}, {np.percentile(scales_x, 97.5):.4f}]")

        print(f"\nScaling factor Y:")
        print(f"  Mean: {np.mean(scales_y):.4f}, Std: {np.std(scales_y):.4f}")
        print(f"  Median: {np.median(scales_y):.4f}")
        print(f"  Range: [{np.min(scales_y):.4f}, {np.max(scales_y):.4f}]")
        print(f"  95% CI: [{np.percentile(scales_y, 2.5):.4f}, {np.percentile(scales_y, 97.5):.4f}]")

        print(f"\nJS divergence improvement:")
        print(f"  Average improvement: {np.mean(improvements):.2f}%")
        print(f"  Max improvement: {np.max(improvements):.2f}%, Min improvement: {np.min(improvements):.2f}%")
        print(f"  Share with improvement >10%: {np.mean(improvements > 10) * 100:.1f}%")

        print(f"\nJS divergence statistics:")
        print(f"  Original mean JS: {np.mean(js_original):.6f}")
        print(f"  Optimized mean JS: {np.mean(js_overall):.6f}")
        print(f"  Mean reduction: {np.mean(js_original - js_overall):.6f}")

        # 可视化
        self._plot_batch_results(results)

    def _plot_batch_results(self, results):
        """Visualize batch results"""
        scales_x = np.array([r['scale_x'] for r in results])
        scales_y = np.array([r['scale_y'] for r in results])
        improvements = np.array([r['improvement'] for r in results])

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. X方向缩放因子分布
        axes[0, 0].hist(scales_x, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(x=np.mean(scales_x), color='red', linestyle='--',
                           label=f'Mean: {np.mean(scales_x):.3f}')
        axes[0, 0].axvline(x=1.0, color='green', linestyle=':', label='scale = 1.0')
        axes[0, 0].set_xlabel('Scale factor (X)')
        axes[0, 0].set_ylabel('Frames')
        axes[0, 0].set_title('Distribution of X scale factors')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Y方向缩放因子分布
        axes[0, 1].hist(scales_y, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(x=np.mean(scales_y), color='red', linestyle='--',
                           label=f'Mean: {np.mean(scales_y):.3f}')
        axes[0, 1].axvline(x=1.0, color='blue', linestyle=':', label='scale = 1.0')
        axes[0, 1].set_xlabel('Scale factor (Y)')
        axes[0, 1].set_ylabel('Frames')
        axes[0, 1].set_title('Distribution of Y scale factors')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 改进比例分布
        axes[1, 0].hist(improvements, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(x=np.mean(improvements), color='red', linestyle='--',
                           label=f'Mean improvement: {np.mean(improvements):.1f}%')
        axes[1, 0].axvline(x=0, color='gray', linestyle=':', label='No improvement')
        axes[1, 0].set_xlabel('JS divergence improvement (%)')
        axes[1, 0].set_ylabel('Frames')
        axes[1, 0].set_title('Distribution of improvement ratio')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. X vs Y缩放因子散点图
        axes[1, 1].scatter(scales_x, scales_y, alpha=0.5, s=20)

        # 添加对角线
        min_val = min(np.min(scales_x), np.min(scales_y))
        max_val = max(np.max(scales_x), np.max(scales_y))
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--',
                        label='y=x', linewidth=2, alpha=0.7)

        axes[1, 1].set_xlabel('Scale factor (X)')
        axes[1, 1].set_ylabel('Scale factor (Y)')
        axes[1, 1].set_title('X vs Y scale factors')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Batch results of optical-flow scaling optimization', fontsize=14)
        plt.tight_layout()
        plt.savefig('batch_optimization_results.png', dpi=150, bbox_inches='tight')
        plt.show()


# 使用示例
def main():
    """
    Main function: demonstrate the full pipeline
    """
    # ====================================================
    # Configuration
    # ====================================================
    SOURCE_FOLDER = "/home/bhzhang/Documents/code/EventDiffusion/data/RAFT_flow_dataset_simple/test/optical_flow"  # 源域光流文件夹
    TARGET_FOLDER = "/home/bhzhang/Documents/assets/bdd100k_70thresh/70images_thresh/flow"  # 目标域光流文件夹
    OUTPUT_FOLDER = ""  # 输出文件夹

    # Optional: saved source distribution (if already built)
    SAVED_SOURCE_DIST = None  # "/path/to/saved/source/distribution"

    # ====================================================
    # Step 1: build or load source distribution
    # ====================================================
    print("Initializing flow scaler...")
    scaler = FlowDistributionScaler(
        n_samples_per_image=500,  # sample 500 points per source image
        n_kde_components=1000  # use 1000 kernels for KDE
    )

    if SAVED_SOURCE_DIST and os.path.exists(SAVED_SOURCE_DIST):
        # Load saved source distribution
        print("Loading saved source distribution...")
        scaler.load_source_distribution(SAVED_SOURCE_DIST)
    else:
        # Build source distribution
        print("Building source distribution...")
        scaler.build_source_distribution(SOURCE_FOLDER, grid_points=500)

        # Save source distribution (optional)
        if OUTPUT_FOLDER:
            source_dist_dir = os.path.join(OUTPUT_FOLDER, 'source_distribution')
            scaler.save_source_distribution(source_dist_dir)

    # ====================================================
    # Step 2: process a single target image (demo)
    # ====================================================
    print("\n" + "=" * 70)
    print("Demo: process a single target image")
    print("=" * 70)

    # 获取一个目标域文件
    target_files = sorted(glob.glob(os.path.join(TARGET_FOLDER, "*.npy")))
    if target_files:
        demo_file = target_files[0]
        print(f"Demo file: {os.path.basename(demo_file)}")

        # 加载并优化
        target_flow = np.load(demo_file)
        result = scaler.optimize_scale_for_single_image(target_flow)

        # 应用缩放
        scaled_flow = scaler.apply_scaling_to_flow(
            target_flow, result['scale_x'], result['scale_y']
        )

        # 打印结果
        print("\n缩放前后对比:")
        print(f"  Original flow X mean: {np.mean(target_flow[0]):.4f}, Y mean: {np.mean(target_flow[1]):.4f}")
        print(f"  Scaled flow X mean: {np.mean(scaled_flow[0]):.4f}, Y mean: {np.mean(scaled_flow[1]):.4f}")
        print(f"  Source X mean: {scaler.source_stats['x_mean']:.4f}, Y mean: {scaler.source_stats['y_mean']:.4f}")

    # ====================================================
    # Step 3: batch process target folder (full processing)
    # ====================================================
    print("\n" + "=" * 70)
    print("Batch processing target folder")
    print("=" * 70)

    all_results = scaler.process_target_folder(
        target_folder=TARGET_FOLDER,
        output_folder=OUTPUT_FOLDER,
        save_scaled_flows=True,
        save_results=True
    )

    # ====================================================
    # Step 4: save final configuration (for inference)
    # ====================================================
    if OUTPUT_FOLDER:
        config = {
            'source_stats': scaler.source_stats,
            'average_scale_x': float(np.mean([r['scale_x'] for r in all_results])),
            'average_scale_y': float(np.mean([r['scale_y'] for r in all_results])),
            'median_scale_x': float(np.median([r['scale_x'] for r in all_results])),
            'median_scale_y': float(np.median([r['scale_y'] for r in all_results])),
            'n_target_files': len(all_results),
            'average_improvement': float(np.mean([r['improvement'] for r in all_results]))
        }

        config_file = os.path.join(OUTPUT_FOLDER, 'scaling_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nConfig saved to: {config_file}")
        print("Config content:")
        for key, value in config.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    np.random.seed(42)

    # 运行主函数
    main()
