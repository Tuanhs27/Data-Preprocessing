import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

def bai1_housing():
    df = pd.read_csv('ITA105_Lab_2_Housing.csv')
    print("--- BÀI 1: HOUSING ---")
    print(df.shape)
    print(df.isnull().sum())
    print(df.describe())

    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df[['dien_tich', 'gia', 'so_phong']])
    plt.savefig('b1_housing_boxplot_before.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='dien_tich', y='gia', data=df)
    plt.savefig('b1_housing_scatter.png')
    plt.close()

    Q1 = df[['dien_tich', 'gia']].quantile(0.25)
    Q3 = df[['dien_tich', 'gia']].quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = ((df[['dien_tich', 'gia']] < (Q1 - 1.5 * IQR)) | (df[['dien_tich', 'gia']] > (Q3 + 1.5 * IQR))).sum()
    print("Outliers IQR:\n", iqr_outliers)

    z_scores = np.abs(zscore(df[['dien_tich', 'gia']]))
    z_outliers = (z_scores > 3).sum()
    print("Outliers Z-score:\n", z_outliers)

    df_cleaned = df[(z_scores < 3).all(axis=1)]
    
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df_cleaned[['dien_tich', 'gia', 'so_phong']])
    plt.savefig('b1_housing_boxplot_after.png')
    plt.close()

def bai2_iot():
    df = pd.read_csv('ITA105_Lab_2_Iot.csv')
    print("--- BÀI 2: IOT ---")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    print(df.isnull().sum())

    plt.figure(figsize=(12, 5))
    for sensor in df['sensor_id'].unique():
        sensor_data = df[df['sensor_id'] == sensor]
        plt.plot(sensor_data.index, sensor_data['temperature'], label=sensor)
    plt.legend()
    plt.savefig('b2_iot_lineplot.png')
    plt.close()

    rolling_outliers_count = 0
    df['temp_zscore'] = np.nan
    
    for sensor in df['sensor_id'].unique():
        mask = df['sensor_id'] == sensor
        rolling_mean = df[mask]['temperature'].rolling(window=10).mean()
        rolling_std = df[mask]['temperature'].rolling(window=10).std()
        
        is_outlier = np.abs(df[mask]['temperature'] - rolling_mean) > (3 * rolling_std)
        rolling_outliers_count += is_outlier.sum()
        
        df.loc[mask, 'temp_zscore'] = np.abs(zscore(df.loc[mask, 'temperature']))

    print("Rolling Mean Outliers:", rolling_outliers_count)
    z_outliers_count = (df['temp_zscore'] > 3).sum()
    print("Z-score Outliers:", z_outliers_count)

    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df[['temperature', 'pressure', 'humidity']])
    plt.savefig('b2_iot_boxplot.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='temperature', y='pressure', data=df)
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='pressure', y='humidity', data=df)
    plt.savefig('b2_iot_scatter.png')
    plt.close()

    df['temperature'] = df['temperature'].clip(lower=df['temperature'].quantile(0.01), upper=df['temperature'].quantile(0.99))
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df[['temperature', 'pressure', 'humidity']])
    plt.savefig('b2_iot_boxplot_after.png')
    plt.close()

def bai3_ecommerce():
    df = pd.read_csv('ITA105_Lab_2_Ecommerce.csv')
    print("--- BÀI 3: ECOMMERCE ---")
    print(df.isnull().sum())
    print(df.describe())

    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df[['price', 'quantity', 'rating']])
    plt.savefig('b3_ecommerce_boxplot.png')
    plt.close()

    numeric_cols = ['price', 'quantity', 'rating']
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    print("IQR Outliers:\n", ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).sum())

    z_scores = np.abs(zscore(df[numeric_cols]))
    print("Z-score Outliers:\n", (z_scores > 3).sum())

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='price', y='quantity', data=df)
    plt.savefig('b3_ecommerce_scatter.png')
    plt.close()

    df = df[(df['price'] > 0) & (df['rating'] <= 5)]
    df['price'] = np.log1p(df['price'])

    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df[['price', 'quantity', 'rating']])
    plt.savefig('b3_ecommerce_boxplot_after.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='price', y='quantity', data=df)
    plt.savefig('b3_ecommerce_scatter_after.png')
    plt.close()

def bai4_multivariate():
    print("--- BÀI 4: MULTIVARIATE OUTLIERS ---")
    
    df_housing = pd.read_csv('ITA105_Lab_2_Housing.csv')
    df_housing['z_dien_tich'] = np.abs(zscore(df_housing['dien_tich']))
    df_housing['z_gia'] = np.abs(zscore(df_housing['gia']))
    housing_outliers = df_housing[(df_housing['z_dien_tich'] > 3) | (df_housing['z_gia'] > 3)]
    
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='dien_tich', y='gia', data=df_housing, color='blue')
    sns.scatterplot(x='dien_tich', y='gia', data=housing_outliers, color='red')
    plt.savefig('b4_housing_multivariate.png')
    plt.close()

    df_iot = pd.read_csv('ITA105_Lab_2_Iot.csv')
    df_iot['z_temp'] = np.abs(zscore(df_iot['temperature']))
    df_iot['z_press'] = np.abs(zscore(df_iot['pressure']))
    iot_outliers = df_iot[(df_iot['z_temp'] > 3) | (df_iot['z_press'] > 3)]

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='temperature', y='pressure', data=df_iot, color='blue')
    sns.scatterplot(x='temperature', y='pressure', data=iot_outliers, color='red')
    plt.savefig('b4_iot_multivariate.png')
    plt.close()

    df_ecom = pd.read_csv('ITA105_Lab_2_Ecommerce.csv')
    df_ecom['z_price'] = np.abs(zscore(df_ecom['price']))
    df_ecom['z_qty'] = np.abs(zscore(df_ecom['quantity']))
    df_ecom['z_rating'] = np.abs(zscore(df_ecom['rating']))
    ecom_outliers = df_ecom[(df_ecom['z_price'] > 3) | (df_ecom['z_qty'] > 3) | (df_ecom['z_rating'] > 3)]

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='price', y='quantity', data=df_ecom, color='blue')
    sns.scatterplot(x='price', y='quantity', data=ecom_outliers, color='red')
    plt.savefig('b4_ecommerce_multivariate.png')
    plt.close()

def main():
    bai1_housing()
    bai2_iot()
    bai3_ecommerce()
    bai4_multivariate()
    print("Hoàn thành phân tích và xuất toàn bộ biểu đồ!")

if __name__ == "__main__":
    main()