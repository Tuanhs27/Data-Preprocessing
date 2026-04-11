import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def yeu_cau_3_outliers_skew(df):
    print("--- YÊU CẦU 3: OUTLIERS & SKEW ---")
    df_out = df.copy()
    num_cols = ['dien_tich', 'gia'] 
    num_cols = [c for c in num_cols if c in df_out.columns]

    for col in num_cols:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_iqr = ((df_out[col] < lower_bound) | (df_out[col] > upper_bound)).sum()
        print(f"{col} - Outliers (IQR): {outliers_iqr}")

        z_scores = np.abs(zscore(df_out[col].dropna()))
        outliers_z = (z_scores > 3).sum()
        print(f"{col} - Outliers (Z-score): {outliers_z}")

        df_out[col] = np.where(df_out[col] > upper_bound, upper_bound, 
                      np.where(df_out[col] < lower_bound, lower_bound, df_out[col]))
        
        df_out[f'{col}_log'] = np.log1p(df_out[col])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df[col], kde=True, ax=axes[0]).set_title(f'Goc - {col}')
        sns.histplot(df_out[f'{col}_log'], kde=True, ax=axes[1]).set_title(f'Log Transform - {col}')
        plt.savefig(f'yc3_outlier_skew_{col}.png')
        plt.close()

    return df_out

def yeu_cau_4_chuan_hoa_ma_hoa(df):
    print("\n--- YÊU CẦU 4: CHUẨN HÓA & MÃ HÓA ---")
    df_norm = df.copy()
    
    num_cols = df_norm.select_dtypes(include=[np.number]).columns
    scaler_z = StandardScaler()
    scaler_mm = MinMaxScaler()
    
    for col in num_cols:
        if '_log' not in col: 
            df_norm[f'{col}_zscore'] = scaler_z.fit_transform(df_norm[[col]])
            df_norm[f'{col}_minmax'] = scaler_mm.fit_transform(df_norm[[col]])
        
    cat_cols = ['vi_tri', 'tinh_trang']
    cat_cols = [c for c in cat_cols if c in df_norm.columns]
    
    df_norm = pd.get_dummies(df_norm, columns=cat_cols, drop_first=True)
    
    print(df_norm.shape)
    return df_norm

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def yeu_cau_5_xu_ly_text(df):
    print("\n--- YÊU CẦU 5: XỬ LÝ TEXT ---")
    df_text = df.copy()
    
    if 'mo_ta' not in df_text.columns:
        mau_cau = ["nhà đẹp rộng rãi", "gần trung tâm tiện đi lại", "nhà mới xây nội thất cơ bản", "cần bán gấp giá rẻ", "view sông cực thoáng mát"]
        df_text['mo_ta'] = [mau_cau[i % len(mau_cau)] for i in range(len(df_text))]
        
    df_text['mo_ta_clean'] = df_text['mo_ta'].apply(preprocess_text)
    
    tfidf = TfidfVectorizer(max_features=50)
    tfidf_matrix = tfidf.fit_transform(df_text['mo_ta_clean'])
    
    print(f"Kich thuoc TF-IDF Matrix: {tfidf_matrix.shape}")
    
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()])
    df_text = pd.concat([df_text, tfidf_df], axis=1)
    
    return df_text

def main():
    try:
        df = pd.read_csv('dataset_assignment_cleaned.csv')
        
        df_yc3 = yeu_cau_3_outliers_skew(df)
        df_yc4 = yeu_cau_4_chuan_hoa_ma_hoa(df_yc3)
        df_yc5 = yeu_cau_5_xu_ly_text(df_yc4)
        
        df_yc5.to_csv('dataset_assignment_phase1_final.csv', index=False)
        print("\nHoan tat Giai doan 1! Da luu file: dataset_assignment_phase1_final.csv")
        
    except FileNotFoundError:
        print("Loi: Khong tim thay file dataset_assignment_cleaned.csv. Vui long chay code Yeu cau 1 & 2 truoc.")

if __name__ == "__main__":
    main()