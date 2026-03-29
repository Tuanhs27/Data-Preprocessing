import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def plot_comparison(df_origin, df_mm, df_z, col_name, save_name, plot_type='hist'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if plot_type == 'hist':
        sns.histplot(df_origin[col_name], kde=True, ax=axes[0]).set_title(f'Gốc - {col_name}')
        sns.histplot(df_mm[col_name], kde=True, ax=axes[1]).set_title('Min-Max Scaling')
        sns.histplot(df_z[col_name], kde=True, ax=axes[2]).set_title('Z-Score Norm')
    elif plot_type == 'box':
        sns.boxplot(y=df_origin[col_name], ax=axes[0]).set_title(f'Gốc - {col_name}')
        sns.boxplot(y=df_mm[col_name], ax=axes[1]).set_title('Min-Max Scaling')
        sns.boxplot(y=df_z[col_name], ax=axes[2]).set_title('Z-Score Norm')
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()

def bai1_sports():
    print("--- BÀI 1: THÔNG SỐ VẬN ĐỘNG VIÊN ---")
    df = pd.read_csv('ITA105_Lab_3_Sports.csv')
    print("Missing values:\n", df.isnull().sum())
    
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df)
    plt.title("Boxplot Gốc - Sports")
    plt.savefig('b1_sports_boxplot_goc.png')
    plt.close()

    scaler_mm = MinMaxScaler()
    scaler_z = StandardScaler()
    
    df_mm = pd.DataFrame(scaler_mm.fit_transform(df), columns=df.columns)
    df_z = pd.DataFrame(scaler_z.fit_transform(df), columns=df.columns)
    
    plot_comparison(df, df_mm, df_z, 'chieu_cao_cm', 'b1_sports_compare_chieucao.png')

def bai2_health():
    print("\n--- BÀI 2: CHỈ SỐ BỆNH NHÂN ---")
    df = pd.read_csv('ITA105_Lab_3_Health.csv')
    
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df)
    plt.title("Boxplot Gốc - Health")
    plt.savefig('b2_health_boxplot_goc.png')
    plt.close()

    scaler_mm = MinMaxScaler()
    scaler_z = StandardScaler()
    
    df_mm = pd.DataFrame(scaler_mm.fit_transform(df), columns=df.columns)
    df_z = pd.DataFrame(scaler_z.fit_transform(df), columns=df.columns)
    
    plot_comparison(df, df_mm, df_z, 'huyet_ap_mmHg', 'b2_health_compare_huyetap.png', 'box')
    
    print(">> Nhận xét Bài 2:")
    print("- Các biến bị ảnh hưởng nhiều bởi ngoại lệ (như huyết áp cực đoan) khi dùng Min-Max sẽ bị nén lại vào một khoảng rất hẹp, khiến dữ liệu bình thường khó phân biệt.")
    print("- Phương pháp Z-Score phù hợp hơn vì nó không giới hạn khoảng giá trị cứng [0,1], giúp giữ nguyên hình dáng phân phối và xử lý tốt dữ liệu có ngoại lệ.")

def bai3_finance():
    print("\n--- BÀI 3: CHỈ SỐ CÔNG TY ---")
    df = pd.read_csv('ITA105_Lab_3_Finance.csv')
    
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df[['doanh_thu_musd', 'loi_nhuan_musd']])
    plt.title("Boxplot Gốc - Finance")
    plt.savefig('b3_finance_boxplot_goc.png')
    plt.close()

    scaler_mm = MinMaxScaler()
    scaler_z = StandardScaler()
    
    df_mm = pd.DataFrame(scaler_mm.fit_transform(df), columns=df.columns)
    df_z = pd.DataFrame(scaler_z.fit_transform(df), columns=df.columns)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.scatterplot(x='doanh_thu_musd', y='loi_nhuan_musd', data=df, ax=axes[0]).set_title('Gốc')
    sns.scatterplot(x='doanh_thu_musd', y='loi_nhuan_musd', data=df_mm, ax=axes[1]).set_title('Min-Max')
    sns.scatterplot(x='doanh_thu_musd', y='loi_nhuan_musd', data=df_z, ax=axes[2]).set_title('Z-Score')
    plt.tight_layout()
    plt.savefig('b3_finance_scatter.png')
    plt.close()

    print(">> Nhận xét Bài 3:")
    print("- Min-Max KHÔNG phù hợp cho tập dữ liệu có các công ty cực lớn (ngoại lệ khổng lồ) vì nó sẽ gán điểm cực trị bằng 1, ép tất cả công ty bình thường về sát mức 0.")
    print("- Linear Regression và KNN rất nhạy cảm với khoảng cách. Nên chọn Z-Score Standardization để thuật toán hội tụ nhanh hơn và không bị chệch hướng bởi ngoại lệ.")

def bai4_gaming():
    print("\n--- BÀI 4: NGƯỜI CHƠI TRỰC TUYẾN ---")
    df = pd.read_csv('ITA105_Lab_3_Gaming.csv')
    
    scaler_mm = MinMaxScaler()
    scaler_z = StandardScaler()
    
    df_mm = pd.DataFrame(scaler_mm.fit_transform(df), columns=df.columns)
    df_z = pd.DataFrame(scaler_z.fit_transform(df), columns=df.columns)
    
    plot_comparison(df, df_mm, df_z, 'gio_choi', 'b4_gaming_compare_giochoi.png', 'hist')

def main():
    bai1_sports()
    bai2_health()
    bai3_finance()
    bai4_gaming()
    print("\nHoàn tất! Đã lưu tất cả các biểu đồ so sánh vào thư mục hiện tại.")


main()