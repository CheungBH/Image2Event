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
    光流分布缩放器：通过优化JS散度为每张目标域图片找到最优缩放因子
    分别优化X和Y方向，范围[0.1, 10]，从1.0开始优化
    """

    def __init__(self, source_folder=None, n_samples_per_image=1000, n_kde_components=1000):
        """
        初始化缩放器

        参数:
        -----------
        source_folder : str, optional
            源域文件夹路径。如果提供，则立即构建源域分布
        n_samples_per_image : int
            每张源域图片的抽样点数
        n_kde_components : int
            KDE核数量
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
        分层抽样：从单张光流图中抽样

        参数:
        -----------
        flow : numpy array, shape (2, H, W)
            光流图
        n_samples : int
            抽样点数

        返回:
        -----------
        x_samples, y_samples : numpy arrays
            X和Y方向的抽样值
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
        从抽样数据构建KDE分布

        参数:
        -----------
        samples : numpy array
            抽样数据
        n_components : int, optional
            KDE核数量。如果为None，使用self.n_kde_components

        返回:
        -----------
        kde_centers : numpy array
            KDE核中心
        kde_weights : numpy array
            核权重（均匀）
        bandwidth : float
            核带宽
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
        将KDE分布离散化到网格上

        参数:
        -----------
        kde_centers : numpy array
            KDE核中心
        kde_weights : numpy array
            核权重
        bandwidth : float
            核带宽
        grid_points : int
            网格点数

        返回:
        -----------
        grid : numpy array
            网格点
        prob_mass : numpy array
            每个网格点的概率质量
        """
        # 确定网格范围（基于核中心，稍微扩展）
        x_min, x_max = np.min(kde_centers), np.max(kde_centers)
        range_extension = (x_max - x_min) * 0.1  # 扩展10%
        x_min -= range_extension
        x_max += range_extension

        # 创建均匀网格
        grid = np.linspace(x_min, x_max, grid_points)
        dx = (x_max - x_min) / (grid_points - 1)

        # 计算KDE在网格上的密度
        density = np.zeros_like(grid, dtype=float)
        sqrt_2pi = np.sqrt(2 * np.pi)

        for center, weight in zip(kde_centers, kde_weights):
            # 高斯核贡献
            gaussian = np.exp(-0.5 * ((grid - center) / bandwidth) ** 2)
            gaussian /= (sqrt_2pi * bandwidth)
            density += weight * gaussian

        # 转换为概率质量（乘以dx）并归一化
        prob_mass = density * dx
        prob_mass /= np.sum(prob_mass)

        return grid, prob_mass, dx

    def build_source_distribution(self, source_folder, grid_points=1000):
        """
        构建源域分布（离线处理）

        参数:
        -----------
        source_folder : str
            源域文件夹路径
        grid_points : int
            离散化网格点数
        """
        print("=" * 70)
        print("构建源域分布")
        print("=" * 70)

        # 获取源域文件列表
        source_files = sorted(glob.glob(os.path.join(source_folder, "*.npy")))
        if not source_files:
            raise ValueError(f"源域文件夹 {source_folder} 中没有找到npy文件")

        print(f"找到 {len(source_files)} 个源域文件")
        print(f"每张图抽样 {self.n_samples_per_image} 个点")

        # 收集所有抽样点
        all_x_samples = []
        all_y_samples = []

        start_time = time.time()

        for i, file_path in enumerate(tqdm(source_files, desc="处理源域文件")):
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
                print(f"已处理 {i + 1}/{len(source_files)} 个文件，用时 {elapsed:.1f}秒")

        all_x_samples = np.array(all_x_samples)
        all_y_samples = np.array(all_y_samples)

        print(f"X方向抽样完成: {len(all_x_samples)} 个点")
        print(f"Y方向抽样完成: {len(all_y_samples)} 个点")

        # 保存源域统计信息
        self.source_stats = {
            'x_mean': np.mean(all_x_samples),
            'x_std': np.std(all_x_samples),
            'x_median': np.median(all_x_samples),
            'y_mean': np.mean(all_y_samples),
            'y_std': np.std(all_y_samples),
            'y_median': np.median(all_y_samples),
            'n_samples_x': len(all_x_samples),
            'n_samples_y': len(all_y_samples)
        }

        print("\n源域统计信息:")
        print(f"  X方向: 均值={self.source_stats['x_mean']:.4f}, "
              f"标准差={self.source_stats['x_std']:.4f}, "
              f"中位数={self.source_stats['x_median']:.4f}")
        print(f"  Y方向: 均值={self.source_stats['y_mean']:.4f}, "
              f"标准差={self.source_stats['y_std']:.4f}, "
              f"中位数={self.source_stats['y_median']:.4f}")

        # 构建KDE分布
        print("\n构建KDE分布...")
        kde_centers_x, kde_weights_x, bandwidth_x = self.build_kde_from_samples(all_x_samples)
        kde_centers_y, kde_weights_y, bandwidth_y = self.build_kde_from_samples(all_y_samples)

        print(f"KDE参数: X方向 {len(kde_centers_x)} 个核, 带宽={bandwidth_x:.4f}")
        print(f"         Y方向 {len(kde_centers_y)} 个核, 带宽={bandwidth_y:.4f}")

        # 离散化到网格
        print("\n离散化KDE到网格...")
        self.grid_x, self.prob_mass_x, self.dx = self.kde_density_on_grid(
            kde_centers_x, kde_weights_x, bandwidth_x, grid_points
        )
        self.grid_y, self.prob_mass_y, self.dy = self.kde_density_on_grid(
            kde_centers_y, kde_weights_y, bandwidth_y, grid_points
        )

        print(f"网格参数: X方向 {len(self.grid_x)} 个点, dx={self.dx:.6f}")
        print(f"         Y方向 {len(self.grid_y)} 个点, dy={self.dy:.6f}")

        elapsed = time.time() - start_time
        print(f"\n源域分布构建完成，总用时 {elapsed:.1f}秒")
        print("=" * 70)

    def save_source_distribution(self, save_dir):
        """
        保存源域分布到文件

        参数:
        -----------
        save_dir : str
            保存目录
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
            json.dump(self.source_stats, f, indent=2)

        print(f"源域分布已保存到: {save_dir}")

    def load_source_distribution(self, load_dir):
        """
        从文件加载源域分布

        参数:
        -----------
        load_dir : str
            加载目录
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

        print(f"源域分布已从 {load_dir} 加载")

    def compute_js_divergence(self, source_prob_mass, target_data, scale, grid, dx):
        """
        计算JS散度

        参数:
        -----------
        source_prob_mass : numpy array
            源域分布的概率质量（离散化）
        target_data : numpy array
            目标域数据
        scale : float
            缩放因子
        grid : numpy array
            网格点
        dx : float
            网格间距

        返回:
        -----------
        js : float
            JS散度值
        """
        # 缩放目标数据
        scaled_target = target_data * scale

        # 计算目标数据在网格上的直方图
        target_hist, _ = np.histogram(
            scaled_target,
            bins=len(grid),
            range=(grid[0], grid[-1]),
            density=True
        )

        # 将密度转换为概率质量（乘以dx）并归一化
        target_prob_mass = target_hist * dx
        target_prob_mass_sum = np.sum(target_prob_mass)

        if target_prob_mass_sum > 0:
            target_prob_mass /= target_prob_mass_sum
        else:
            # 如果所有点都在网格外，返回一个很大的值
            return 1.0

        # 避免零值
        epsilon = 1e-10
        p = np.clip(source_prob_mass, epsilon, None)
        q = np.clip(target_prob_mass, epsilon, None)

        # 计算JS散度
        m = 0.5 * (p + q)
        js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))

        return js

    def optimize_scale_for_single_image(self, target_flow, max_target_samples=10000):
        """
        为单张目标域图片优化缩放因子

        参数:
        -----------
        target_flow : numpy array, shape (2, H, W)
            目标域光流图
        max_target_samples : int
            目标数据最大抽样点数（加速优化）

        返回:
        -----------
        result : dict
            优化结果，包含缩放因子、JS散度等信息
        """
        # 提取X和Y方向数据并展平
        target_x = target_flow[0].flatten()
        target_y = target_flow[1].flatten()

        # 如果数据量太大，进行随机抽样以加速优化
        if len(target_x) > max_target_samples:
            indices = np.random.choice(len(target_x), max_target_samples, replace=False)
            target_x = target_x[indices]
            target_y = target_y[indices]

        print(f"目标数据: {len(target_x)} 个点")

        # 分别优化X和Y方向
        results = {'x': None, 'y': None}

        for direction, target_data, grid, prob_mass, dx in [
            ('x', target_x, self.grid_x, self.prob_mass_x, self.dx),
            ('y', target_y, self.grid_y, self.prob_mass_y, self.dy)
        ]:
            print(f"\n优化 {direction.upper()} 方向...")

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
                print(f"  最优缩放因子: {result.x:.6f}")
                print(f"  最小JS散度: {result.fun:.6f}")
                print(f"  迭代次数: {result.nfev}, 用时: {elapsed:.3f}秒")
            else:
                print(f"  {direction.upper()}方向优化失败: {result.message}")
                # 使用均值比作为后备方案
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
                print(f"  使用后备缩放因子: {fallback_scale:.6f}")

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
            print("优化结果汇总:")
            print("=" * 50)
            print(f"X方向缩放因子: {result_summary['scale_x']:.6f}")
            print(f"Y方向缩放因子: {result_summary['scale_y']:.6f}")
            print(f"总体JS散度: 原始={result_summary['js_original']:.6f}, "
                  f"优化后={result_summary['js_overall']:.6f}")
            print(f"改进: {result_summary['improvement']:.2f}%")
            print(f"总优化时间: {result_summary['time_x'] + result_summary['time_y']:.3f}秒")

            return result_summary
        else:
            raise ValueError("优化失败")

    def apply_scaling_to_flow(self, flow, scale_x, scale_y):
        """
        将缩放因子应用到光流图

        参数:
        -----------
        flow : numpy array, shape (2, H, W)
            原始光流图
        scale_x, scale_y : float
            X和Y方向的缩放因子

        返回:
        -----------
        scaled_flow : numpy array
            缩放后的光流图
        """
        scaled_flow = flow.copy()
        scaled_flow[0] *= scale_x
        scaled_flow[1] *= scale_y
        return scaled_flow

    def process_target_folder(self, target_folder, output_folder=None,
                              save_scaled_flows=True, save_results=True):
        """
        批量处理目标域文件夹中的所有图片

        参数:
        -----------
        target_folder : str
            目标域文件夹路径
        output_folder : str, optional
            输出文件夹路径
        save_scaled_flows : bool
            是否保存缩放后的光流图
        save_results : bool
            是否保存优化结果

        返回:
        -----------
        all_results : list
            所有图片的优化结果
        """
        # 获取目标域文件列表
        target_files = sorted(glob.glob(os.path.join(target_folder, "*.npy")))
        if not target_files:
            raise ValueError(f"目标域文件夹 {target_folder} 中没有找到npy文件")

        print("=" * 70)
        print(f"处理目标域文件夹: {target_folder}")
        print(f"找到 {len(target_files)} 个文件")
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

        for i, file_path in enumerate(tqdm(target_files, desc="处理目标域文件")):
            print(f"\n处理文件 {i + 1}/{len(target_files)}: {os.path.basename(file_path)}")

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

            print(f"\n优化结果已保存到: {results_file}")

        # 分析统计结果
        self._analyze_batch_results(all_results)

        total_time = time.time() - total_start_time
        print(f"\n批量处理完成!")
        print(f"总文件数: {len(target_files)}")
        print(f"成功处理: {len(all_results)}")
        print(f"总用时: {total_time:.1f}秒")
        print(f"平均每帧: {total_time / len(all_results):.3f}秒")
        print("=" * 70)

        return all_results

    def _analyze_batch_results(self, results):
        """分析批量处理结果"""
        if not results:
            print("没有结果可分析")
            return

        scales_x = np.array([r['scale_x'] for r in results])
        scales_y = np.array([r['scale_y'] for r in results])
        improvements = np.array([r['improvement'] for r in results])
        js_overall = np.array([r['js_overall'] for r in results])
        js_original = np.array([r['js_original'] for r in results])

        print("\n" + "=" * 70)
        print("批量处理结果统计")
        print("=" * 70)

        print(f"X方向缩放因子:")
        print(f"  均值: {np.mean(scales_x):.4f}, 标准差: {np.std(scales_x):.4f}")
        print(f"  中位数: {np.median(scales_x):.4f}")
        print(f"  范围: [{np.min(scales_x):.4f}, {np.max(scales_x):.4f}]")
        print(f"  95%区间: [{np.percentile(scales_x, 2.5):.4f}, {np.percentile(scales_x, 97.5):.4f}]")

        print(f"\nY方向缩放因子:")
        print(f"  均值: {np.mean(scales_y):.4f}, 标准差: {np.std(scales_y):.4f}")
        print(f"  中位数: {np.median(scales_y):.4f}")
        print(f"  范围: [{np.min(scales_y):.4f}, {np.max(scales_y):.4f}]")
        print(f"  95%区间: [{np.percentile(scales_y, 2.5):.4f}, {np.percentile(scales_y, 97.5):.4f}]")

        print(f"\nJS散度改进:")
        print(f"  平均改进: {np.mean(improvements):.2f}%")
        print(f"  最大改进: {np.max(improvements):.2f}%, 最小改进: {np.min(improvements):.2f}%")
        print(f"  改进>10%的比例: {np.mean(improvements > 10) * 100:.1f}%")

        print(f"\nJS散度统计:")
        print(f"  原始平均JS: {np.mean(js_original):.6f}")
        print(f"  优化后平均JS: {np.mean(js_overall):.6f}")
        print(f"  平均降低: {np.mean(js_original - js_overall):.6f}")

        # 可视化
        self._plot_batch_results(results)

    def _plot_batch_results(self, results):
        """可视化批量处理结果"""
        scales_x = np.array([r['scale_x'] for r in results])
        scales_y = np.array([r['scale_y'] for r in results])
        improvements = np.array([r['improvement'] for r in results])

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. X方向缩放因子分布
        axes[0, 0].hist(scales_x, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(x=np.mean(scales_x), color='red', linestyle='--',
                           label=f'均值: {np.mean(scales_x):.3f}')
        axes[0, 0].axvline(x=1.0, color='green', linestyle=':', label='scale=1.0')
        axes[0, 0].set_xlabel('X方向缩放因子')
        axes[0, 0].set_ylabel('帧数')
        axes[0, 0].set_title('X方向缩放因子分布')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Y方向缩放因子分布
        axes[0, 1].hist(scales_y, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(x=np.mean(scales_y), color='red', linestyle='--',
                           label=f'均值: {np.mean(scales_y):.3f}')
        axes[0, 1].axvline(x=1.0, color='blue', linestyle=':', label='scale=1.0')
        axes[0, 1].set_xlabel('Y方向缩放因子')
        axes[0, 1].set_ylabel('帧数')
        axes[0, 1].set_title('Y方向缩放因子分布')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 改进比例分布
        axes[1, 0].hist(improvements, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(x=np.mean(improvements), color='red', linestyle='--',
                           label=f'平均改进: {np.mean(improvements):.1f}%')
        axes[1, 0].axvline(x=0, color='gray', linestyle=':', label='无改进')
        axes[1, 0].set_xlabel('JS散度改进 (%)')
        axes[1, 0].set_ylabel('帧数')
        axes[1, 0].set_title('分布改进比例')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. X vs Y缩放因子散点图
        axes[1, 1].scatter(scales_x, scales_y, alpha=0.5, s=20)

        # 添加对角线
        min_val = min(np.min(scales_x), np.min(scales_y))
        max_val = max(np.max(scales_x), np.max(scales_y))
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--',
                        label='y=x', linewidth=2, alpha=0.7)

        axes[1, 1].set_xlabel('X方向缩放因子')
        axes[1, 1].set_ylabel('Y方向缩放因子')
        axes[1, 1].set_title('X vs Y缩放因子')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('光流缩放优化批量处理结果', fontsize=14)
        plt.tight_layout()
        plt.savefig('batch_optimization_results.png', dpi=150, bbox_inches='tight')
        plt.show()


# 使用示例
def main():
    """
    主函数：演示完整流程
    """
    # ====================================================
    # 配置参数
    # ====================================================
    SOURCE_FOLDER = "/home/bhzhang/Documents/code/EventDiffusion/data/RAFT_flow_dataset_simple/test/optical_flow"  # 源域光流文件夹
    TARGET_FOLDER = "/home/bhzhang/Documents/assets/bdd100k_70thresh/70images_thresh/flow"  # 目标域光流文件夹
    OUTPUT_FOLDER = ""  # 输出文件夹

    # 可选：已保存的源域分布文件（如果已经构建过）
    SAVED_SOURCE_DIST = None  # "/path/to/saved/source/distribution"

    # ====================================================
    # 步骤1：构建或加载源域分布
    # ====================================================
    print("初始化光流缩放器...")
    scaler = FlowDistributionScaler(
        n_samples_per_image=500,  # 每张源域图抽样500个点
        n_kde_components=1000  # KDE使用1000个核
    )

    if SAVED_SOURCE_DIST and os.path.exists(SAVED_SOURCE_DIST):
        # 加载已保存的源域分布
        print("加载已保存的源域分布...")
        scaler.load_source_distribution(SAVED_SOURCE_DIST)
    else:
        # 构建源域分布
        print("构建源域分布...")
        scaler.build_source_distribution(SOURCE_FOLDER, grid_points=500)

        # 保存源域分布（可选）
        if OUTPUT_FOLDER:
            source_dist_dir = os.path.join(OUTPUT_FOLDER, 'source_distribution')
            scaler.save_source_distribution(source_dist_dir)

    # ====================================================
    # 步骤2：处理单个目标域图片（演示）
    # ====================================================
    print("\n" + "=" * 70)
    print("演示：处理单个目标域图片")
    print("=" * 70)

    # 获取一个目标域文件
    target_files = sorted(glob.glob(os.path.join(TARGET_FOLDER, "*.npy")))
    if target_files:
        demo_file = target_files[0]
        print(f"演示文件: {os.path.basename(demo_file)}")

        # 加载并优化
        target_flow = np.load(demo_file)
        result = scaler.optimize_scale_for_single_image(target_flow)

        # 应用缩放
        scaled_flow = scaler.apply_scaling_to_flow(
            target_flow, result['scale_x'], result['scale_y']
        )

        # 打印结果
        print("\n缩放前后对比:")
        print(f"  原始光流 X均值: {np.mean(target_flow[0]):.4f}, Y均值: {np.mean(target_flow[1]):.4f}")
        print(f"  缩放后光流 X均值: {np.mean(scaled_flow[0]):.4f}, Y均值: {np.mean(scaled_flow[1]):.4f}")
        print(f"  源域 X均值: {scaler.source_stats['x_mean']:.4f}, Y均值: {scaler.source_stats['y_mean']:.4f}")

    # ====================================================
    # 步骤3：批量处理目标域文件夹（完整处理）
    # ====================================================
    print("\n" + "=" * 70)
    print("批量处理目标域文件夹")
    print("=" * 70)

    all_results = scaler.process_target_folder(
        target_folder=TARGET_FOLDER,
        output_folder=OUTPUT_FOLDER,
        save_scaled_flows=True,
        save_results=True
    )

    # ====================================================
    # 步骤4：保存最终配置（用于推理时）
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

        print(f"\n配置已保存到: {config_file}")
        print("配置内容:")
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