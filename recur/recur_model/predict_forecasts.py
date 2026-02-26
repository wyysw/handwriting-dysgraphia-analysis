import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os
from typing import Optional, Dict, Any, Tuple
import warnings
import matplotlib
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['font.family'] = ['Arial']  # 或者你系统中的其他中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class MonteCarloPredictor:
    def __init__(self, models_dir='./outputs'):
        self.models_dir = models_dir
        self.models = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.categorical_columns = []
        self.best_model_name = None
        
    def load_models_and_metadata(self):
        """加载训练好的模型和元数据"""
        print("Loading models and metadata...")
        
        # 加载元数据
        metadata_path = os.path.join(self.models_dir, 'model_metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.feature_columns = metadata['feature_columns']
        self.categorical_columns = metadata['categorical_columns']
        
        # 加载标签编码器
        encoders_path = os.path.join(self.models_dir, 'label_encoders.joblib')
        self.label_encoders = joblib.load(encoders_path)
        
        # 加载所有模型
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_model.joblib')]
        for model_file in model_files:
            model_name = model_file.replace('_model.joblib', '')
            model_path = os.path.join(self.models_dir, model_file)
            self.models[model_name] = joblib.load(model_path)
            print(f"Loaded {model_name} model")
    
    def extract_features(self, child_data_row: pd.Series) -> np.ndarray:
        """从儿童数据行提取特征向量"""
        features = []
        
        for col in self.feature_columns:
            if col in child_data_row.index:
                value = child_data_row[col]
                
                # 处理类别特征
                if col in self.categorical_columns and col in self.label_encoders:
                    # 如果是字符串，进行编码
                    if isinstance(value, str):
                        try:
                            value = self.label_encoders[col].transform([value])[0]
                        except ValueError:
                            # 如果遇到未见过的类别，使用最常见的类别
                            value = 0
                features.append(value)
            else:
                # 如果特征缺失，使用0填充
                features.append(0)
        
        return np.array(features)
    
    def monte_carlo_forecast(self, 
                           child_data: pd.DataFrame,
                           child_id: str,
                           n_steps: int = 6,
                           n_simulations: int = 100,
                           time_interval: float = 0.5,
                           model_name: Optional[str] = None,
                           base_noise_std: float = 0.05,
                           noise_growth_rate: float = 0.3,
                           model_uncertainty_factor: float = 0.1) -> Dict[str, Any]:
        """
        蒙特卡洛模拟预测
        
        Args:
            child_data: 儿童数据
            child_id: 儿童ID
            n_steps: 预测步数
            n_simulations: 模拟次数
            time_interval: 时间间隔（年）
            model_name: 使用的模型名称
            noise_std: 噪声标准差
        """
        
        model = self.models[model_name]
        
        # 选择特定儿童的数据
        child_data_filtered = child_data[child_data['身份证号'] == child_id].copy()
        
        if len(child_data_filtered) == 0:
            raise ValueError(f"No data found for child_id: {child_id}")
        
        # 获取第一行数据（每个儿童应该只有一行）
        if len(child_data_filtered) > 1:
            print(f"Warning: Multiple rows found for child_id {child_id}, using the first row")
        
        child_row = child_data_filtered.iloc[0]
        
        # 获取最后一个数据点
    
        last_age = child_row['target_age']
        last_target = child_row['target_sph_r']
        
        # 提取历史数据 (t1 到 t7)
        historical_ages = []
        historical_targets = []
        
        for i in range(1, 8):  # t1 到 t7
            age_col = f'age_t{i}'
            sph_col = f'sph_r_t{i}'
            
            if age_col in child_row.index and sph_col in child_row.index:
                age_val = child_row[age_col]
                sph_val = child_row[sph_col]
                
                # 只保留非空值
                if pd.notna(age_val) and pd.notna(sph_val):
                    historical_ages.append(float(age_val))
                    historical_targets.append(float(sph_val))
        
        # 转换为numpy数组并按年龄排序
        if historical_ages:
            historical_data = list(zip(historical_ages, historical_targets))
            historical_data.sort(key=lambda x: x[0])  # 按年龄排序
            historical_ages = np.array([x[0] for x in historical_data])
            historical_targets = np.array([x[1] for x in historical_data])
        else:
            historical_ages = np.array([])
            historical_targets = np.array([])
        
        print(f"Starting prediction for child {child_id}")
        print(f"Child info: 性别={child_row.get('性别', 'N/A')}, 矫正方式={child_row.get('矫正方式', 'N/A')}, 筛查区={child_row.get('筛查区', 'N/A')}")
        print(f"Historical data points (t1-t7): {len(historical_ages)}")
        if len(historical_ages) > 0:
            print(f"Historical age range: {historical_ages.min():.2f} - {historical_ages.max():.2f}")
            print(f"Historical target range: {historical_targets.min():.4f} - {historical_targets.max():.4f}")
        print(f"Current target age: {last_age}, Current target: {last_target}")
        
        # 提取初始特征
        initial_features = self.extract_features(child_row)
        
        print(f"Feature vector shape: {initial_features.shape}")
        print(f"Feature columns: {self.feature_columns}")
        
        # 找到关键特征在特征向量中的位置
        age_idx = None
        target_idx = None
        
        if 'target_age' in self.feature_columns:
            age_idx = self.feature_columns.index('target_age')
        if 'target_sph_r' in self.feature_columns:  
            target_idx = self.feature_columns.index('target_sph_r')
        
        # 找到历史特征的索引
        age_indices = []      # age_t1 到 age_t7 的索引
        sph_r_indices = []    # sph_r_t1 到 sph_r_t7 的索引     

        for i in range(1, 8):
            age_col = f'age_t{i}'
            sph_col = f'sph_r_t{i}'
            
            if age_col in self.feature_columns:
                age_indices.append(self.feature_columns.index(age_col))
            if sph_col in self.feature_columns:
                sph_r_indices.append(self.feature_columns.index(sph_col))
        
        print(f"Found {len(age_indices)} age indices and {len(sph_r_indices)} sph_r indices")
        print(f"Target age index: {age_idx}, Target sph index: {target_idx}")
        # 存储所有模拟轨迹
        all_predictions = np.zeros((n_simulations, n_steps))
        
        # 蒙特卡洛模拟
        for sim in range(n_simulations):
            # 每次模拟开始时重置特征
            sim_features = np.copy(initial_features)
            pred_trajectory = []
            
            # 循环预测未来每一步
            for step in range(n_steps):
                # 用当前特征预测下一步目标值
                pred_val = model.predict(sim_features.reshape(1, -1))[0]
                
                current_noise_std = base_noise_std * (1 + noise_growth_rate * step)
            
            # 模型不确定性也随时间增加
                model_uncertainty = model_uncertainty_factor * (1 + 0.5 * step)
                # 添加随机噪声以模拟不确定性
                if model_uncertainty > 0 and sim > 0:  # 第一个模拟保持确定性作为基准
                   model_noise = np.random.normal(0, model_uncertainty)
                   pred_val += model_noise
                if current_noise_std > 0:
                # 观测噪声随时间和模拟次数变化
                   obs_noise = np.random.normal(0, current_noise_std)
                
                   # 为不同的模拟添加不同程度的变异性
                   simulation_factor = 1 + 0.1 * (sim / n_simulations)  # 后面的模拟更多变异
                   obs_noise *= simulation_factor
                
                   pred_val += obs_noise
              
                
                pred_trajectory.append(pred_val)
                
                # === 滑动窗口更新逻辑 ===
                # 1. 更新当前年龄和目标值
                # new_age = last_age + time_interval * (step )
                new_age = last_age + time_interval * (step + 1)

                if age_idx is not None:
                    sim_features[age_idx] = new_age
                if target_idx is not None:
                    sim_features[target_idx] = pred_val
                
                # 2. 滑动窗口：更新历史序列
                if len(age_indices) > 0 and len(sph_r_indices) > 0:
                    # 获取当前的历史序列
                    current_ages = [sim_features[idx] for idx in age_indices]
                    current_targets = [sim_features[idx] for idx in sph_r_indices]
                    
                    # 创建新的历史序列：移除最老的，加入最新的
                    new_ages = current_ages[1:] + [new_age]
                    new_targets = current_targets[1:] + [pred_val]
                    
                    # 更新特征向量中的历史序列
                    for i, idx in enumerate(age_indices):
                        if i < len(new_ages):
                            sim_features[idx] = new_ages[i]
                    
                    for i, idx in enumerate(sph_r_indices):
                        if i < len(new_targets):
                            sim_features[idx] = new_targets[i]
                
                # 调试信息（仅显示第一次模拟的前几步）
                if sim == 0 and step < 3:
                    print(f"\nSimulation {sim}, Step {step}:")
                    print(f"Predicted value: {pred_val:.4f}")
                    print(f"New age: {new_age:.2f}")
            
            # 保存这次模拟的完整轨迹
            all_predictions[sim, :] = np.array(pred_trajectory)
        
        # 计算统计量
        mean_pred = np.mean(all_predictions, axis=0)
        median_pred = np.median(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)
        
        # 计算百分位数区间
        lower_5 = np.percentile(all_predictions, 5, axis=0)
        upper_95 = np.percentile(all_predictions, 95, axis=0)
        lower_25 = np.percentile(all_predictions, 25, axis=0)
        upper_75 = np.percentile(all_predictions, 75, axis=0)
        
        # 计算标准差区间
        lower_1std = mean_pred - std_pred
        upper_1std = mean_pred + std_pred
        lower_2std = mean_pred - 2 * std_pred
        upper_2std = mean_pred + 2 * std_pred
        
        return {
            'child_id': child_id,
            'model_name': model_name,
            'last_age': last_age,
            'last_target': last_target,
            'historical_ages': historical_ages,
            'historical_targets': historical_targets,
            'time_points': [last_age + time_interval * (i +1 ) for i in range(n_steps)],
            'mean_pred': mean_pred,
            'median_pred': median_pred,
            'std_pred': std_pred,
            'lower_5': lower_5,
            'upper_95': upper_95,
            'lower_25': lower_25,
            'upper_75': upper_75,
            'lower_1std': lower_1std,
            'upper_1std': upper_1std,
            'lower_2std': lower_2std,
            'upper_2std': upper_2std,
            'all_predictions': all_predictions,
            'n_simulations': n_simulations,
            'n_steps': n_steps
        }
    
    def plot_forecast(self, forecast_result: Dict[str, Any], 
                     save_path: Optional[str] = None,
                     show_plot: bool = True) -> None:
        """绘制预测结果（包含历史数据 t1-t7）"""
        child_id = forecast_result['child_id']
        model_name = forecast_result['model_name']
        last_age = forecast_result['last_age']
        last_target = forecast_result['last_target']
        
        # 历史数据 (t1-t7)
        historical_ages = forecast_result['historical_ages']
        historical_targets = forecast_result['historical_targets']
        
        # 预测数据
        time_points = forecast_result['time_points']
        mean_pred = forecast_result['mean_pred']
        median_pred = forecast_result['median_pred']
        
        plt.figure(figsize=(14, 8))
        
        # 绘制历史数据点 (t1-t7)
        # if len(historical_ages) > 0:
        #     plt.plot(historical_ages, historical_targets, 'ko-',
        #             linewidth=2, markersize=6, label='历史数据 (t1-t7)', alpha=0.8)
        #
        # # 绘制当前目标点 (target_age, target_sph_r)
        # plt.scatter(last_age, last_target, color='red', s=120, zorder=10,
        #            label='当前目标时点', marker='s', edgecolors='darkred', linewidth=2)
        #
        # 如果有历史数据，连接最后的历史点到当前目标点
        # if len(historical_ages) > 0:
        #     plt.plot([historical_ages[-1], last_age], [historical_targets[-1], last_target],
        #             'k--', alpha=0.6, linewidth=1.5, label='历史到目标连线')
        #

        # 修改后（正确绘图）：
        # 将目标点加入历史数据
        full_historical_ages = np.append(historical_ages, last_age)
        full_historical_targets = np.append(historical_targets, last_target)

        # 绘制完整历史数据（包括目标点）
        plt.plot(full_historical_ages, full_historical_targets, 'ko-',
                 linewidth=2, markersize=6, label='历史数据', alpha=0.8)



        # 绘制预测中位数（主要预测线）
        plt.plot(time_points, median_pred, 'b-o', linewidth=3, markersize=8, 
                label='预测中位数', alpha=0.8)
        
        # 绘制预测均值（辅助线）
        plt.plot(time_points, mean_pred, 'g--', linewidth=2, 
                label='预测均值', alpha=0.7)
        
        # 绘制90%预测区间（最外层阴影）
        plt.fill_between(time_points, forecast_result['lower_5'], 
                        forecast_result['upper_95'], 
                        color='lightblue', alpha=0.3, label='90%预测区间')
        
        # 绘制50%预测区间（中间阴影）
        plt.fill_between(time_points, forecast_result['lower_25'], 
                        forecast_result['upper_75'], 
                        color='skyblue', alpha=0.5, label='50%预测区间')
        
        # 绘制±1标准差区间
        plt.fill_between(time_points, forecast_result['lower_1std'], 
                        forecast_result['upper_1std'], 
                        color='yellow', alpha=0.3, label='±1标准差')
        
        # 添加目标点到预测的连接线
        plt.plot([last_age, time_points[0]], [last_target, median_pred[0]], 
                'b--', alpha=0.5, linewidth=1)
        
        # 添加分割线（历史/预测分界）
        plt.axvline(x=last_age + 0.05, color='gray', linestyle='--', 
                   alpha=0.7, label='目标/预测分界')
        
        # 美化图表
        plt.xlabel('年龄', fontsize=12)
        plt.ylabel('屈光度 (sph_r)', fontsize=12)
        plt.title(f'儿童视力发展轨迹：历史检查 → 目标时点 → 未来预测\n'
                 f'ID: {child_id} | 模型: {model_name} | '
                 f'历史点数: {len(historical_ages)} | 模拟次数: {forecast_result["n_simulations"]}', 
                 fontsize=14, fontweight='bold')
        
        # 设置图例
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.7)
        
        # 设置坐标轴范围（包含所有数据）
        all_ages = []
        all_values = []
        
        if len(historical_ages) > 0:
            all_ages.extend(historical_ages)
            all_values.extend(historical_targets)
        
        all_ages.extend([last_age])
        all_ages.extend(time_points)
        all_values.extend([last_target])
        all_values.extend(forecast_result['lower_2std'])
        all_values.extend(forecast_result['upper_2std'])
        
        # X轴范围
        x_margin = (np.max(all_ages) - np.min(all_ages)) * 0.05
        plt.xlim(np.min(all_ages) - x_margin, np.max(all_ages) + x_margin)
        
        # Y轴范围
        y_margin = (np.max(all_values) - np.min(all_values)) * 0.1
        plt.ylim(np.min(all_values) - y_margin, np.max(all_values) + y_margin)
        
        # 添加文本说明
        info_text = f'目标时点: {last_age:.1f} 岁 (sph_r: {last_target:.3f})\n'
        if len(historical_ages) > 0:
            info_text += f'历史数据: {historical_ages.min():.1f} - {historical_ages.max():.1f} 岁\n'
        info_text += f'预测范围: {time_points[0]:.1f} - {time_points[-1]:.1f} 岁'
        
        plt.text(0.02, 0.98, info_text, 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Forecast plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
    
    def run_prediction(self, 
                      data_path: str,
                      child_id: str,
                      n_steps: int = 6,
                      n_simulations: int = 100,
                      time_interval: float = 0.5,
                      model_name: Optional[str] = None) -> Dict[str, Any]:
        """运行完整的预测流程"""
        # 加载模型和元数据
        self.load_models_and_metadata()
        
        # 加载数据
        data = pd.read_csv(data_path)

        # 执行预测
        forecast_result = self.monte_carlo_forecast(
            child_data=data,
            child_id=child_id,
            n_steps=n_steps,
            n_simulations=n_simulations,
            time_interval=time_interval,
            model_name=model_name
        )
        
        # 绘制预测结果
        save_path = None
        save_path = os.path.join(self.models_dir, f'forecast_{child_id}_{model_name or self.best_model_name}.png')
        
        self.plot_forecast(forecast_result, save_path)
        
        # 打印预测摘要
        self.print_forecast_summary(forecast_result)
        
        return forecast_result
    
    def print_forecast_summary(self, forecast_result: Dict[str, Any]) -> None:
        """打印预测摘要"""
        print("\n" + "="*60)
        print("预测摘要")
        print("="*60)
        print(f"儿童ID: {forecast_result['child_id']}")
        print(f"使用模型: {forecast_result['model_name']}")
        
        
        # 历史数据信息 (t1-t7)
        historical_ages = forecast_result['historical_ages']
        historical_targets = forecast_result['historical_targets']
        
        print(f"历史数据点数 (t1-t7): {len(historical_ages)}")
        if len(historical_ages) > 0:
            print(f"历史年龄范围: {historical_ages.min():.2f} - {historical_ages.max():.2f} 岁")
            print(f"历史屈光度范围: {historical_targets.min():.4f} - {historical_targets.max():.4f}")
            print("历史数据详情:")
            for i, (age, target) in enumerate(zip(historical_ages, historical_targets)):
                print(f"  t{i+1}: 年龄 {age:.2f}, 屈光度 {target:.4f}")
        
        print(f"当前目标年龄: {forecast_result['last_age']:.2f}")
        print(f"当前目标屈光度: {forecast_result['last_target']:.4f}")
        print(f"预测步数: {forecast_result['n_steps']}")
        print(f"模拟次数: {forecast_result['n_simulations']}")
        print("-" * 60)
        print("未来预测:")
        
        for i, (time_point, median, mean, std) in enumerate(zip(
            forecast_result['time_points'],
            forecast_result['median_pred'],
            forecast_result['mean_pred'],
            forecast_result['std_pred']
        )):
            print(f"步骤 {i+1:2d} | 年龄: {time_point:5.2f} | "
                  f"中位数: {median:7.4f} | 均值: {mean:7.4f} | "
                  f"标准差: {std:6.4f}")
        
        print("="*60)


def main():
    """主函数 - 示例用法"""
    predictor = MonteCarloPredictor()
    
    # 示例参数
    data_path = 'processed_data.csv'
    example_child_id = '110101200706280029'  # 替换为实际的child_id
    
    try:
        forecast_result = predictor.run_prediction(
            data_path=data_path,
            child_id=example_child_id,
            n_steps=6,
            n_simulations=200,
            time_interval=0.5,
            model_name='LightGBM'  # 选择模型
        )
        
        print("预测完成！")
        
    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        print("请检查数据路径和child_id是否正确")

if __name__ == "__main__":
    main()
