import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def yeu_cau_1_kham_pha(df):
    print("--- YÊU CẦU 1: KHÁM PHÁ DỮ LIỆU ---")
    print(df.shape)
    print(df.isnull().sum())
    print(df.duplicated().sum())
    print(df.describe())
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in num_cols:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f'Histogram - {col}')
        
        sns.boxplot(y=df[col], ax=axes[1])
        axes[1].set_title(f'Boxplot - {col}')
        
        sns.violinplot(y=df[col], ax=axes[2])
        axes[2].set_title(f'Violin Plot - {col}')
        
        plt.tight_layout()
        plt.savefig(f'yc1_plot_{col}.png')
        plt.close()
        
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        print(f"\nPhân phối cột {col}:\n", df[col].value_counts().head(10))

def yeu_cau_2_lam_sach(df):
    print("\n--- YÊU CẦU 2: LÀM SẠCH DỮ LIỆU ---")
    df_clean = df.copy()
    
    df_clean = df_clean.drop_duplicates()
    
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    
    for col in num_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
    if 'gia' in df_clean.columns:
        df_clean['gia'] = df_clean['gia'].apply(lambda x: x if x > 0 else df_clean['gia'].median())
        
    if 'so_phong' in df_clean.columns:
        df_clean['so_phong'] = df_clean['so_phong'].apply(lambda x: x if x > 0 else df_clean['so_phong'].median())
        
    for col in cat_cols:
        df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
        
    print(df_clean.shape)
    print(df_clean.isnull().sum())
    
    return df_clean

def main():
    try:
        df = pd.read_csv('dataset_assignment.csv')
        
        yeu_cau_1_kham_pha(df)
        df_cleaned = yeu_cau_2_lam_sach(df)
        
        df_cleaned.to_csv('dataset_assignment_cleaned.csv', index=False)
        print("\nHoàn tất! Đã lưu dữ liệu sạch và biểu đồ.")
        
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file. Vui lòng đổi tên file dữ liệu của bạn thành 'dataset_assignment.csv' hoặc sửa lại đường dẫn trong hàm main().")


main()