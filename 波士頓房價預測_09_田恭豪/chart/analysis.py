import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 獲取當前腳本的目錄路徑
script_dir = os.path.dirname(__file__)

# 構造 train_dataset.csv 的相對路徑
relative_path = os.path.join('..', 'train_dataset.csv')
dataset_path = os.path.abspath(os.path.join(script_dir, relative_path))

# 嘗試讀取 CSV 檔案
try:
    data = pd.read_csv(dataset_path)
    print("成功載入資料")
except FileNotFoundError:
    print(f"找不到檔案: {dataset_path}")
    raise
except Exception as e:
    print(f"發生錯誤: {e}")
    raise

# 計算特徵相關性
featuresCorr = data.corr()

# 定義閾值，這裡可以根據需求進行修改
threshold = 0.51

# 根據閾值選擇特徵
targetCorr = featuresCorr['PRICE']
selectedFeatures = targetCorr[abs(targetCorr) > threshold].drop('PRICE').index.tolist()

# 檢查是否有選擇特徵
if not selectedFeatures:
    print("沒有選擇特徵進行回歸分析")
else:
    print(f"選擇特徵: {selectedFeatures}")

    # 提取特徵和目標變數
    X = data[selectedFeatures]
    y = data['PRICE']

    # 初始化最高準確率及其對應的模型名稱
    max_knn_accuracy = -1
    max_gs_accuracy = -1
    max_dec_accuracy = -1
    max_rf_accuracy = -1  # 新增隨機森林模型準確率變數

    best_knn_model = None  # 儲存最佳KNN模型
    best_rf_model = None   # 儲存最佳隨機森林模型

    mse_lr = mse_knn = mse_gs = mse_dec = mse_rf = 0  # 新增隨機森林MSE
    r2_lr = r2_knn = r2_gs = r2_dec = r2_rf = 0    # 新增隨機森林R^2

    correct_within_tolerance = [0, 0, 0, 0, 0]  # 用於儲存所有模型的正確比率

    # 迭代運行五次並找出最高準確率的一次
    for i in range(5):
        # 分割數據集為訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        # 標準化數據
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 使用K近鄰演算法
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)
        knn_score = knn.score(X_test_scaled, y_test)
        if knn_score > max_knn_accuracy:
            max_knn_accuracy = knn_score
            best_knn_model = knn  # 更新最佳模型

        # 計算KNN模型的MSE和R^2
        mse_knn = mean_squared_error(y_test, knn.predict(X_test_scaled))
        r2_knn = r2_score(y_test, knn.predict(X_test_scaled))

        # 設定KNN回歸模型和GridSearchCV的參數網格
        param_grid = {
            'n_neighbors': [3, 5, 8, 10],
            'weights': ['uniform', 'distance']
        }

        knn_gs = KNeighborsRegressor()
        grid_search = GridSearchCV(knn_gs, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_scaled, y_train)
        gs_score = grid_search.best_estimator_.score(X_test_scaled, y_test)
        if gs_score > max_gs_accuracy:
            max_gs_accuracy = gs_score

        # 計算GridSearchCV模型的MSE和R^2
        mse_gs = mean_squared_error(y_test, grid_search.best_estimator_.predict(X_test_scaled))
        r2_gs = r2_score(y_test, grid_search.best_estimator_.predict(X_test_scaled))

        # 初始化決策樹回歸模型
        dec_tree = DecisionTreeRegressor(random_state=42)
        dec_tree.fit(X_train, y_train)
        dec_score = dec_tree.score(X_test, y_test)
        if dec_score > max_dec_accuracy:
            max_dec_accuracy = dec_score

        # 計算決策樹模型的MSE和R^2
        mse_dec = mean_squared_error(y_test, dec_tree.predict(X_test))
        r2_dec = r2_score(y_test, dec_tree.predict(X_test))

        # 初始化隨機森林回歸模型
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_score = rf_model.score(X_test, y_test)
        if rf_score > max_rf_accuracy:
            max_rf_accuracy = rf_score
            best_rf_model = rf_model  # 更新最佳模型

        # 計算隨機森林模型的MSE和R^2
        mse_rf = mean_squared_error(y_test, rf_model.predict(X_test))
        r2_rf = r2_score(y_test, rf_model.predict(X_test))

        # 確保最佳模型存在
        if best_knn_model is not None:
            # 計算在容忍範圍內的正確比率
            tolerance_percentage = 0.2  # 變更容忍範圍內的正確比率: 20%
            tolerance_threshold = tolerance_percentage * np.abs(y_test)
            absolute_errors_knn = np.abs(best_knn_model.predict(X_test_scaled) - y_test)
            correct_within_tolerance[0] = np.mean(absolute_errors_knn <= tolerance_threshold)

            # 對GridSearchCV模型進行相同的計算
            absolute_errors_gs = np.abs(grid_search.best_estimator_.predict(X_test_scaled) - y_test)
            correct_within_tolerance[1] = np.mean(absolute_errors_gs <= tolerance_threshold)

            # 對決策樹模型進行相同的計算
            absolute_errors_dec = np.abs(dec_tree.predict(X_test) - y_test)
            correct_within_tolerance[2] = np.mean(absolute_errors_dec <= tolerance_threshold)

            # 對隨機森林模型進行相同的計算
            if best_rf_model is not None:
                absolute_errors_rf = np.abs(best_rf_model.predict(X_test) - y_test)
                correct_within_tolerance[3] = np.mean(absolute_errors_rf <= tolerance_threshold)

    # 輸出結果
    print(f"K近鄰模組_準確率：{max_knn_accuracy}")
    print(f"GridSearchCV網格搜索模組_準確率：{max_gs_accuracy}")
    print(f"決策樹分析_準確率：{max_dec_accuracy}")
    print(f"隨機森林回歸_準確率：{max_rf_accuracy}")

# 儲存全局變數供外部使用
max_knn_accuracy = max_knn_accuracy
max_gs_accuracy = max_gs_accuracy
max_dec_accuracy = max_dec_accuracy
max_rf_accuracy = max_rf_accuracy  
correct_within_tolerance = correct_within_tolerance
mse_knn = mse_knn
r2_knn = r2_knn
mse_gs = mse_gs
r2_gs = r2_gs
mse_dec = mse_dec
r2_dec = r2_dec
mse_rf = mse_rf  
r2_rf = r2_rf  
