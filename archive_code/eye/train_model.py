import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from tabpfn import TabPFNRegressor
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import json

class ModelTrainer:
    def __init__(self, output_dir='./outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.label_encoders = {}
        self.feature_columns = None
        self.categorical_columns = ['性别', '筛查区', '矫正方式']
        
    def load_and_preprocess_data(self, input_path):
        """加载和预处理数据"""
        print("Loading and preprocessing data...")
        df = pd.read_csv(input_path)
        
        # 特征矩阵和标签
        X = df.drop(columns=['target_sph_r', '身份证号'])
        y = df['target_sph_r']
        
        # 保存特征列名（用于后续预测）
        self.feature_columns = X.columns.tolist()
        
        # 对类别特征进行标签编码
        for col in self.categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        return X, y
    
    def train_models(self, X, y):
        """训练多个回归模型"""
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 定义不同的回归模型
        models = {
            # "TabPFN": TabPFNRegressor(),
            # "XGBoost": XGBRegressor(random_state=42),
            "LightGBM": LGBMRegressor(random_state=42),
            # "CatBoost": CatBoostRegressor(
            #     learning_rate=0.1,
            #     iterations=1000,
            #     depth=6,
            #     verbose=0,
            #     random_state=42
            # ),
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=100, 
                random_state=42
            )
        }
        
        # 存储评估结果和训练好的模型
        results = []
        trained_models = {}
        
        # 训练并评估每个模型
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 进行预测
            predictions = model.predict(X_test)
            
            # 评估模型
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            print(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            print("-" * 40)
            
            # 保存结果和模型
            results.append([model_name, mse, mae, r2])
            trained_models[model_name] = model
        
        return results, trained_models, X_test, y_test
    
    
    def save_models_and_metadata(self, trained_models, results):
        """保存模型和元数据"""
        # 保存评估结果
        results_df = pd.DataFrame(results, columns=["Model", "MSE", "MAE", "R2"])
        results_df.to_csv(os.path.join(self.output_dir, 'regression_results.csv'), index=False)
        print("\nModel Performance Summary:")
        print(results_df)
        
        # 保存每个训练好的模型
        for model_name, model in trained_models.items():
            model_path = os.path.join(self.output_dir, f'{model_name}_model.joblib')
            joblib.dump(model, model_path)
            print(f"Saved {model_name} model to {model_path}")
        
        # 保存标签编码器和特征信息
        metadata = {
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns
        }
        
        # 保存标签编码器
        encoders_path = os.path.join(self.output_dir, 'label_encoders.joblib')
        joblib.dump(self.label_encoders, encoders_path)
        
        # 保存元数据
        metadata_path = os.path.join(self.output_dir, 'model_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Saved metadata to {metadata_path}")
        print(f"Saved label encoders to {encoders_path}")
    
    def run_training_pipeline(self, input_path='processed_data.csv'):
        """运行完整的训练流程"""
        print("Starting model training pipeline...")
        
        # 加载和预处理数据
        X, y = self.load_and_preprocess_data(input_path)
        
        # 训练模型
        results, trained_models, X_test, y_test = self.train_models(X, y)
        
        # 保存模型和元数据
        self.save_models_and_metadata(trained_models, results)
        
        print("\nTraining pipeline completed successfully!")
        return trained_models, results

def main():
    """主函数"""
    trainer = ModelTrainer()
    trained_models, results = trainer.run_training_pipeline()
    
    # 显示所有模型的性能
    results_df = pd.DataFrame(results, columns=["Model", "MSE", "MAE", "R2"])
    print("\n" + "="*60)
    print("所有模型性能对比：")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)

    
    return trainer, trained_models, results

if __name__ == "__main__":
    main()
