import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def bai1_kham_pha_du_lieu(df):
    print("--- Bài 1 ---")
    print(df.shape)
    print(df.describe())
    print(df.isnull().sum())

def bai2_xu_ly_du_lieu_thieu(df):
    print("--- Bài 2 ---")
    df_filled = df.copy()
    
    if 'Price' in df_filled.columns:
        df_filled['Price'] = df_filled['Price'].fillna(df_filled['Price'].mean())
    if 'StockQuantity' in df_filled.columns:
        df_filled['StockQuantity'] = df_filled['StockQuantity'].fillna(df_filled['StockQuantity'].mean())
    if 'Category' in df_filled.columns:
        df_filled['Category'] = df_filled['Category'].fillna(df_filled['Category'].mode()[0])
        
    df_dropped = df.dropna()
    print(df_filled.shape)
    print(df_dropped.shape)
    
    return df_filled

def bai3_xu_ly_du_lieu_loi(df):
    print("--- Bài 3 ---")
    df_clean = df.copy()
    
    df_clean['Price'] = df_clean['Price'].apply(lambda x: x if x >= 0 else np.nan)
    df_clean['StockQuantity'] = df_clean['StockQuantity'].apply(lambda x: x if x >= 0 else np.nan)
    
    df_clean['Price'] = df_clean['Price'].fillna(df_clean['Price'].median())
    df_clean['StockQuantity'] = df_clean['StockQuantity'].fillna(df_clean['StockQuantity'].median())
    
    df_clean = df_clean[df_clean['Rating'].isin([1, 2, 3, 4, 5])]
    print(df_clean.shape)
    
    return df_clean

def bai4_lam_muot_du_lieu(df):
    print("--- Bài 4 ---")
    df_smooth = df.copy()
    
    df_smooth['Price_MA_3'] = df_smooth['Price'].rolling(window=3, min_periods=1).mean()
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_smooth['Price'].values[:50], label='Original Price', marker='o', alpha=0.5)
    plt.plot(df_smooth['Price_MA_3'].values[:50], label='MA=3', marker='x', linestyle='--')
    plt.legend()
    plt.savefig('bieu_do_bai_4.png')
    plt.close()
    return df_smooth

def bai5_chuan_hoa_du_lieu(df):
    print("--- Bài 5 ---")
    df_norm = df.copy()
    
    if 'Category' in df_norm.columns:
        df_norm['Category'] = df_norm['Category'].astype(str).str.lower()
    if 'Description' in df_norm.columns:
        df_norm['Description'] = df_norm['Description'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
        
    df_norm['Price_VND'] = df_norm['Price'] * 25000
    
    return df_norm
def main():
    df = pd.read_csv('data_lab1.csv')
    
    bai1_kham_pha_du_lieu(df)
    df = bai2_xu_ly_du_lieu_thieu(df)
    df = bai3_xu_ly_du_lieu_loi(df)
    df = bai4_lam_muot_du_lieu(df)
    df = bai5_chuan_hoa_du_lieu(df)
    
    df.to_csv('data_lab1_Cleaned_Final.csv', index=False)
main()