import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageTk
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tkinter as tk
from io import StringIO

# 獲取當前腳本的目錄路徑
# script_dir = os.path.dirname(__file__)

# 構造 train_dataset.csv 的相對路徑
# relative_path = os.path.join('..', 'train_dataset.csv')
# dataset_path = os.path.abspath(os.path.join(script_dir, relative_path))

# 測試用路徑
dataset_path = 'train_dataset.csv'

# 嘗試讀取 CSV 檔案
try:
    data = pd.read_csv(dataset_path)
    print("成功載入資料.")
except FileNotFoundError:
    print(f"找不到檔案: {dataset_path}")
except Exception as e:
    print(f"發生錯誤: {e}")

def plot_missing_values(data):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')  # 隱藏圖表

    # 獲取缺失值資訊並轉換為文本
    buf = StringIO()
    data.info(buf=buf)
    info_str = buf.getvalue()

    # 在圖表上顯示缺失值資訊
    ax.text(0.5, 0.5, info_str, fontsize=12, ha='center', va='center', wrap=True)
    plt.title('Missing Values Check')

    # 將圖表轉換為 PIL Image
    canvas = FigureCanvas(fig)
    canvas.draw()
    pil_image = Image.frombytes('RGBA', canvas.get_width_height(), canvas.buffer_rgba())

    # 調整圖片大小
    pil_image = pil_image.resize((400, 300), Image.LANCZOS)

    plt.close(fig)  # 關閉圖表，釋放資源
    return pil_image

def plot_boxplot(data):
    # 求出四分位距(IQR)=Q3-Q1與上邊界(天花板)和下邊界(地板)
    Q1 = data['PRICE'].quantile(0.25)
    Q3 = data['PRICE'].quantile(0.75)
    IQR = Q3 - Q1
    Upper = Q3 + 1.5 * IQR
    Lower = Q1 - 1.5 * IQR
    print('Summary statistics:')
    print('Q3=', Q3, 'Q1=', Q1, 'IQR=', IQR, 'Upper=', Upper, 'Lower=', Lower)

    # 創建圖表並繪製盒鬚圖
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data['PRICE'], showmeans=True)
    ax.set_title('Boxplot of Price')
    ax.set_ylabel('Price')

    # 將圖表轉換為 PIL Image
    canvas = FigureCanvas(fig)
    canvas.draw()
    pil_image = Image.frombytes('RGBA', canvas.get_width_height(), canvas.buffer_rgba())

    # 調整圖片大小
    pil_image = pil_image.resize((400, 300), Image.LANCZOS)

    plt.close(fig)  # 關閉圖表，釋放資源
    return pil_image

def plot_normal_distribution(data):
    # 創建圖表並繪製常態分佈圖
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data['PRICE'], kde=True, ax=ax)
    ax.set_title('Normal Distribution of Price')
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')

    # 將圖表轉換為 PIL Image
    canvas = FigureCanvas(fig)
    canvas.draw()
    pil_image = Image.frombytes('RGBA', canvas.get_width_height(), canvas.buffer_rgba())

    # 調整圖片大小
    pil_image = pil_image.resize((400, 300), Image.LANCZOS)

    plt.close(fig)  # 關閉圖表，釋放資源
    return pil_image

# 初始化 Tkinter 視窗
root = tk.Tk()
root.title('數據分析圖表')

# 創建一個 Frame 用來容納圖表
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# 創建一個 Frame 用來橫向排列圖表
chart_frame = tk.Frame(frame)
chart_frame.pack(side=tk.TOP, padx=5, pady=5)

# 顯示缺失值圖表
missing_values_image = plot_missing_values(data)
missing_values_photo = ImageTk.PhotoImage(image=missing_values_image)
label_missing_values = tk.Label(chart_frame, image=missing_values_photo)
label_missing_values.pack(side=tk.LEFT, padx=5, pady=5)

# 顯示盒鬚圖
boxplot_image = plot_boxplot(data)
boxplot_photo = ImageTk.PhotoImage(image=boxplot_image)
label_boxplot = tk.Label(chart_frame, image=boxplot_photo)
label_boxplot.pack(side=tk.LEFT, padx=5, pady=5)

# 顯示常態分佈圖
normal_dist_image = plot_normal_distribution(data)
normal_dist_photo = ImageTk.PhotoImage(image=normal_dist_image)
label_normal_distribution = tk.Label(chart_frame, image=normal_dist_photo)
label_normal_distribution.pack(side=tk.LEFT, padx=5, pady=5)

root.mainloop()
