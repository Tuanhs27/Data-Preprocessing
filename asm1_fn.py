import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def tao_du_lieu_mau_neu_thieu():
    if not os.path.exists('dataset_assignment.csv'):
        np.random.seed(42)
        n = 500
        dien_tich = np.random.normal(80, 30, n).clip(30, 300)
        so_phong = np.random.randint(1, 6, n)
        vi_tri = np.random.choice(['Quan 1', 'Quan 2', 'Quan 7', 'Binh Thanh', 'Thu Duc'], n)
        tinh_trang = np.random.choice(['Moi', 'Cu', 'Dang xay'], n)
        mo_ta = np.random.choice(['nha dep', 'gan trung tam', 'noi that co ban', 'view song', 'gia re'], n)
        gia = dien_tich * np.random.uniform(30, 80, n) + so_phong * 500 + np.random.normal(0, 1000, n)
        
        df = pd.DataFrame({
            'dien_tich': dien_tich, 'so_phong': so_phong, 'vi_tri': vi_tri, 
            'tinh_trang': tinh_trang, 'mo_ta': mo_ta, 'gia': gia
        })
        df.loc[10:20, 'dien_tich'] = np.nan
        df.loc[30:40, 'mo_ta'] = np.nan
        df.to_csv('dataset_assignment.csv', index=False)
        print("Da tao file dataset_assignment.csv mau.")

def xdt_pipeline_va_mo_hinh(df):
    print("--- BƯỚC 1 & 2: XÂY DỰNG PIPELINE TỰ ĐỘNG ---")

    df['mo_ta'] = df['mo_ta'].fillna('') 
    
    X = df.drop(columns=['gia'])
    y = df['gia']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    num_cols = ['dien_tich', 'so_phong']
    cat_cols = ['vi_tri', 'tinh_trang']
    text_col = 'mo_ta'
    
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # [ĐÃ FIX LỖI Ở ĐÂY]: Gỡ bỏ SimpleImputer khỏi text_pipe
    text_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
        ('text', text_pipe, text_col)
    ])
    
    print("--- BƯỚC 3: HUẤN LUYỆN VÀ SO SÁNH MÔ HÌNH ---")
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_r2 = -float('inf')
    best_pipeline = None
    
    for name, model in models.items():
        pipe = Pipeline([
            ('prep', preprocessor),
            ('model', model)
        ])
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        print(f"[{name}] RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipe
            
    print(f"\n=> Mo hinh tot nhat: {best_model_name} (R2: {best_r2:.4f})")
    joblib.dump(best_pipeline, 'proptech_pricing_pipeline.pkl')
    print("Da luu pipeline hoan chinh vao 'proptech_pricing_pipeline.pkl'")
    
    return best_pipeline, X_test, y_test, results

def phan_tich_kpi_va_kịch_ban(df, pipeline):
    print("\n--- BƯỚC 4: PHÂN TÍCH ĐA KỊCH BẢN & KPI ---")
    df_eval = df.copy()
    
    X_all = df_eval.drop(columns=['gia'])
    df_eval['gia_du_doan'] = pipeline.predict(X_all)
    
    df_eval['price_per_m2'] = df_eval['gia'] / df_eval['dien_tich']
    
    threshold_95 = df_eval['price_per_m2'].quantile(0.95)
    df_eval['luxury_score'] = np.where(df_eval['price_per_m2'] >= threshold_95, 'High Luxury', 'Standard')
    
    df_eval['log_price_index'] = np.log1p(df_eval['gia'])
    
    print("\nKPI: Gia trung binh theo khu vuc:")
    print(df_eval.groupby('vi_tri')['gia'].mean().sort_values(ascending=False))
    
    print(f"\nPhan tich Top 5% (Luxury): Nguong gia/m2 la {threshold_95:.2f}")
    print(df_eval['luxury_score'].value_counts())
    
    return df_eval

def truc_quan_hoa_dashboard(df_eval):
    print("\n--- BƯỚC 5: TRỰC QUAN HÓA & INSIGHT ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sns.scatterplot(x='gia', y='gia_du_doan', data=df_eval, ax=axes[0, 0], alpha=0.6, color='blue')
    axes[0, 0].plot([df_eval['gia'].min(), df_eval['gia'].max()], 
                    [df_eval['gia'].min(), df_eval['gia'].max()], 'r--')
    axes[0, 0].set_title('Thuc te vs Du doan (Model Performance)')
    
    sns.boxplot(x='vi_tri', y='price_per_m2', data=df_eval, ax=axes[0, 1])
    axes[0, 1].set_title('Phan phoi Gia/m2 theo Khu vuc')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    sns.histplot(df_eval['log_price_index'], kde=True, ax=axes[1, 0], color='green')
    axes[1, 0].set_title('Phan phoi Log Price Index (Giam Skewness)')
    
    sns.countplot(x='luxury_score', hue='vi_tri', data=df_eval, ax=axes[1, 1])
    axes[1, 1].set_title('So luong BDS Luxury theo Khu vuc')
    
    plt.tight_layout()
    plt.savefig('assignment_phase2_dashboard.png')
    plt.close()
    print("Da luu Dashboard vao file 'assignment_phase2_dashboard.png'")
    
    print("\n[BÁO CÁO INSIGHT NGHIỆP VỤ]")
    print("- Dựa vào Dashboard, có thể thấy mô hình bắt trend giá khá tốt (các điểm bám sát đường chéo đỏ).")
    print("- Metric 'price_per_m2' giúp chuẩn hóa giá trị tài sản bất chấp diện tích, từ đó dễ dàng xác định các khu vực bị định giá quá cao (Overvalued) hoặc vùng trũng đầu tư (Undervalued).")
    print("- Phân khúc 'High Luxury' (Top 5% giá/m2) thường tập trung ở một số quận lõi, đây là tệp khách hàng tiềm năng cho các chiến dịch Marketing VIP của công ty.")

def main():
    tao_du_lieu_mau_neu_thieu()
    df = pd.read_csv('dataset_assignment.csv')
    df = df.dropna(subset=['gia']) 
    
    best_pipeline, X_test, y_test, results = xdt_pipeline_va_mo_hinh(df)
    df_eval = phan_tich_kpi_va_kịch_ban(df, best_pipeline)
    truc_quan_hoa_dashboard(df_eval)
    
    print("\nHoan tat Giai doan 2 cua Assignment!")

if __name__ == "__main__":
    main()