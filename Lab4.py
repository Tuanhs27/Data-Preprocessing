# PHẦN 1: Import thư viện
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def main():
    print("--- LAB 3 PHẦN 2: TRAIN MACHINE LEARNING MODEL ---")
    
    # PHẦN 2: Tạo dataset
    data = {
        "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
        "Score": [2, 4, 5, 6, 7, 8, 8.5, 9]
    }
    df = pd.DataFrame(data)
    print("\nDataset ban đầu:")
    print(df)

    # PHẦN 3: Tách Input và Output
    X = df[["Hours"]]  # Input (Feature) bắt buộc phải là mảng 2 chiều (DataFrame)
    y = df["Score"]    # Output (Label) là mảng 1 chiều (Series)

    # PHẦN 4 & 5: Khởi tạo và Train Model
    model = LinearRegression()
    model.fit(X, y)
    print("\nĐã train model xong!")

    # PHẦN 6: Dự đoán 1 giá trị mới (ví dụ: học 5 giờ)
    # Lưu ý: dữ liệu đưa vào dự đoán cũng phải có dạng mảng 2 chiều giống X
    new_hours = pd.DataFrame({"Hours": [5]})
    predicted_score = model.predict(new_hours)
    print(f"\nSinh viên học 5 giờ -> Predicted score: {predicted_score[0]:.2f}")

    # PHẦN 7: Dự đoán nhiều giá trị cùng lúc (ví dụ: 4, 6, 9 giờ)
    new_data = pd.DataFrame({"Hours": [4, 6, 9]})
    predictions = model.predict(new_data)
    print("\nDự đoán cho 4, 6, 9 giờ lần lượt là:")
    for h, p in zip([4, 6, 9], predictions):
        print(f"- {h} giờ -> {p:.2f} điểm")

    # PHẦN 8: Vẽ biểu đồ trực quan (Visualization)
    plt.figure(figsize=(8, 5))
    
    # Vẽ các điểm dữ liệu thực tế
    plt.scatter(X, y, color='blue', label='Dữ liệu thực tế')
    
    # Vẽ đường thẳng dự đoán của model (Regression Line)
    plt.plot(X, model.predict(X), color='red', label='Đường dự đoán')
    
    # Gắn nhãn và tiêu đề
    plt.xlabel("Hours studied")
    plt.ylabel("Score")
    plt.title("Hours vs Score")
    plt.legend()
    
    # Lưu biểu đồ thành file ảnh
    plt.savefig('Lab3_Phan2_Regression.png')
    plt.close()
    print("\nĐã hoàn tất! Biểu đồ được lưu thành file 'Lab3_Phan2_Regression.png'.")

if __name__ == "__main__":
    main()