import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import glob
import os


class DistributionAnalyzer:
    """分布分析器，支持多种相似性度量"""

    def __init__(self, folder1, folder2):
        self.folder1 = folder1
        self.folder2 = folder2
        self.data1 = self._load_data(folder1)
        self.data2 = self._load_data(folder2)

        print(f"数据1: {self.data1.shape} 个元素，均值={np.mean(self.data1):.4f}，标准差={np.std(self.data1):.4f}")
        print(f"数据2: {self.data2.shape} 个元素，均值={np.mean(self.data2):.4f}，标准差={np.std(self.data2):.4f}")

    def _load_data(self, folder):
        """加载文件夹中所有npy文件"""
        files = sorted(glob.glob(os.path.join(folder, "*.npy")))
        arrays = []
        for f in files:
            arr = np.load(f)  # (2, H, W)
            # 处理两个通道
            for channel in range(arr.shape[0]):
                arrays.append(arr[channel].flatten())
        return np.concatenate(arrays)

    def compute_histograms(self, scale=1.0, bins=100):
        """计算缩放后的直方图"""
        scaled_data1 = self.data1 * scale

        # 统一bin范围
        all_data = np.concatenate([scaled_data1, self.data2])
        hist_range = (np.min(all_data), np.max(all_data))

        # 计算直方图（概率密度）
        hist1, bin_edges = np.histogram(scaled_data1, bins=bins,
                                        range=hist_range, density=True)
        hist2, _ = np.histogram(self.data2, bins=bins,
                                range=hist_range, density=True)

        # 计算bin中心（用于绘图）
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return hist1, hist2, bin_centers, bin_edges

    def kl_divergence_hist(self, scale=1.0, bins=100):
        """直方图KL散度"""
        hist1, hist2, _, _ = self.compute_histograms(scale, bins)

        epsilon = 1e-10
        hist1_safe = np.clip(hist1, epsilon, None)
        hist2_safe = np.clip(hist2, epsilon, None)

        kl = np.sum(hist1_safe * np.log(hist1_safe / hist2_safe))
        return kl

    def js_divergence_hist(self, scale=1.0, bins=100):
        """直方图JS散度"""
        hist1, hist2, _, _ = self.compute_histograms(scale, bins)

        epsilon = 1e-10
        hist1_safe = np.clip(hist1, epsilon, None)
        hist2_safe = np.clip(hist2, epsilon, None)

        # 中间分布
        m = 0.5 * (hist1_safe + hist2_safe)

        # JS散度
        js = 0.5 * np.sum(hist1_safe * np.log(hist1_safe / m)) + \
             0.5 * np.sum(hist2_safe * np.log(hist2_safe / m))

        # 归一化到[0,1]（使用自然对数时除以ln(2)）
        js_normalized = js / np.log(2)

        return js_normalized

    def pearson_correlation(self, scale=1.0):
        """皮尔逊相关系数"""
        scaled = self.data1 * scale
        return np.corrcoef(scaled, self.data2)[0, 1]

    def mse(self, scale=1.0):
        """均方误差"""
        scaled = self.data1 * scale
        return np.mean((scaled - self.data2) ** 2)

    def find_optimal_scale(self, metric='js', bins=100):
        """寻找最优缩放因子"""

        if metric == 'kl':
            def objective(scale):
                return self.kl_divergence_hist(scale, bins)
        elif metric == 'js':
            def objective(scale):
                return self.js_divergence_hist(scale, bins)
        elif metric == 'corr':
            def objective(scale):
                return -self.pearson_correlation(scale)  # 负号因为要最大化
        elif metric == 'mse':
            def objective(scale):
                return self.mse(scale)
        else:
            raise ValueError(f"未知度量: {metric}")

        # 搜索范围（基于均值比）
        mean_ratio = np.mean(self.data2) / np.mean(self.data1) if np.mean(self.data1) != 0 else 1.0
        lower = max(0.001, mean_ratio * 0.1)
        upper = min(1000, mean_ratio * 10)

        result = minimize_scalar(objective, bounds=(lower, upper), method='bounded')

        return {
            'optimal_scale': result.x,
            'min_value': result.fun if metric != 'corr' else -result.fun,
            'metric': metric
        }

    def analyze_all_metrics(self, bins=100):
        """使用所有度量方法分析"""
        metrics = ['kl', 'js', 'corr', 'mse']
        results = {}

        print("\n" + "=" * 60)
        print("不同度量方法的最优缩放因子:")
        print("-" * 60)

        for metric in metrics:
            try:
                result = self.find_optimal_scale(metric, bins)
                results[metric] = result

                if metric == 'corr':
                    print(f"{metric:4s}: 缩放因子 = {result['optimal_scale']:8.4f}, "
                          f"最大相关系数 = {result['min_value']:.6f}")
                else:
                    print(f"{metric:4s}: 缩放因子 = {result['optimal_scale']:8.4f}, "
                          f"最小{metric.upper()} = {result['min_value']:.6f}")
            except Exception as e:
                print(f"{metric:4s}: 计算失败 - {e}")

        return results

    def plot_comparison(self, optimal_scales=None, bins=100):
        """可视化比较"""
        if optimal_scales is None:
            optimal_scales = {'mse': 1.0, 'js': 1.0, 'corr': 1.0}

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 原始数据分布
        axes[0, 0].hist(self.data1, bins=bins, alpha=0.5, label='原始数据1', density=True)
        axes[0, 0].hist(self.data2, bins=bins, alpha=0.5, label='数据2', density=True)
        axes[0, 0].set_title('原始数据分布')
        axes[0, 0].legend()
        axes[0, 0].set_xlabel('值')
        axes[0, 0].set_ylabel('概率密度')

        # 2. 不同缩放因子下的分布对比
        colors = ['blue', 'green', 'red']
        for idx, (metric, scale) in enumerate(optimal_scales.items()):
            hist1, hist2, bin_centers, _ = self.compute_histograms(scale, bins)

            axes[0, 1].plot(bin_centers, hist1, color=colors[idx], alpha=0.7,
                            label=f'数据1 × {scale:.3f} ({metric})', linewidth=2)

        hist1_orig, hist2_orig, bin_centers, _ = self.compute_histograms(1.0, bins)
        axes[0, 1].plot(bin_centers, hist2_orig, 'k--', label='数据2 (目标)', linewidth=3)
        axes[0, 1].set_title('不同缩放因子下的分布对比')
        axes[0, 1].legend()
        axes[0, 1].set_xlabel('值')
        axes[0, 1].set_ylabel('概率密度')

        # 3. 不同度量随缩放因子的变化
        scales = np.logspace(-1, 1, 100)  # 0.1到10

        # KL散度
        kl_values = [self.kl_divergence_hist(s, bins) for s in scales]
        axes[0, 2].semilogx(scales, kl_values, 'b-', label='KL散度')

        # JS散度
        js_values = [self.js_divergence_hist(s, bins) for s in scales]
        axes[0, 2].semilogx(scales, js_values, 'g-', label='JS散度')

        axes[0, 2].set_xlabel('缩放因子 (log scale)')
        axes[0, 2].set_ylabel('散度值')
        axes[0, 2].set_title('散度度量 vs 缩放因子')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. 相关系数随缩放因子的变化
        corr_values = [self.pearson_correlation(s) for s in scales]
        axes[1, 0].semilogx(scales, corr_values, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('缩放因子 (log scale)')
        axes[1, 0].set_ylabel('皮尔逊相关系数')
        axes[1, 0].set_title('相关系数 vs 缩放因子')
        axes[1, 0].grid(True, alpha=0.3)

        # 5. MSE随缩放因子的变化
        mse_values = [self.mse(s) for s in scales]
        axes[1, 1].semilogx(scales, mse_values, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('缩放因子 (log scale)')
        axes[1, 1].set_ylabel('均方误差 (MSE)')
        axes[1, 1].set_title('MSE vs 缩放因子')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. 最优缩放因子对比
        metrics = list(optimal_scales.keys())
        scales_vals = list(optimal_scales.values())

        axes[1, 2].bar(range(len(metrics)), scales_vals, color=['blue', 'green', 'red'])
        axes[1, 2].set_xticks(range(len(metrics)))
        axes[1, 2].set_xticklabels(metrics)
        axes[1, 2].set_ylabel('最优缩放因子')
        axes[1, 2].set_title('不同度量的最优缩放因子')
        axes[1, 2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('distribution_comparison_all_metrics.png', dpi=150, bbox_inches='tight')
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 初始化分析器
    analyzer = DistributionAnalyzer("/home/bhzhang/Documents/code/Image2Event/assets/DSEC_RAFT_single/flow",
                                    "/home/bhzhang/Documents/code/EventDiffusion/data/RAFT_flow_dataset_simple/test/optical_flow")

    # 分析所有度量方法
    results = analyzer.analyze_all_metrics(bins=200)

    # 提取最优缩放因子用于绘图
    optimal_scales = {}
    for metric, result in results.items():
        optimal_scales[metric] = result['optimal_scale']

    # 可视化
    analyzer.plot_comparison(optimal_scales, bins=200)

    # 输出总结
    print("\n" + "=" * 60)
    print("分析总结:")
    print("-" * 60)

    mean_ratio = np.mean(analyzer.data2) / np.mean(analyzer.data1)
    print(f"简单均值比: {mean_ratio:.6f}")

    for metric, result in results.items():
        if metric == 'corr':
            print(f"{metric.upper():3s}: 缩放因子 = {result['optimal_scale']:.6f}, "
                  f"最大相关系数 = {result['min_value']:.6f}")
        else:
            print(f"{metric.upper():3s}: 缩放因子 = {result['optimal_scale']:.6f}, "
                  f"最小值 = {result['min_value']:.6f}")