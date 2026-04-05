import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import re

warnings.filterwarnings('ignore')

def preprocess_text(text):
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    stop_words = ['và', 'nhưng', 'rất', 'khá', 'hơi', 'chưa', 'có']
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def bai1_hotel_reviews():
    print("\n--- BÀI 1: REVIEW KHÁCH SẠN ---")
    df = pd.read_csv('ITA105_Lab_4_Hotel_reviews.csv')
    df = df.dropna()
    
    le = LabelEncoder()
    df['hotel_name_encoded'] = le.fit_transform(df['hotel_name'])
    df['customer_type_encoded'] = le.fit_transform(df['customer_type'])
    
    df['tokens'] = df['review_text'].apply(preprocess_text)
    df['cleaned_text'] = df['tokens'].apply(lambda x: ' '.join(x))
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
    print(tfidf_matrix.shape)
    
    w2v_model = Word2Vec(sentences=df['tokens'], vector_size=10, window=3, min_count=1, workers=4)
    if 'sạch' in w2v_model.wv.key_to_index:
        print(w2v_model.wv.most_similar('sạch', topn=3))
    
    print("TF-IDF: Phân loại văn bản, trích xuất từ khóa.")
    print("Word2Vec: Phân tích ngữ nghĩa, tìm từ đồng nghĩa.")

def bai2_match_comments():
    print("\n--- BÀI 2: BÌNH LUẬN TRẬN ĐẤU ---")
    df = pd.read_csv('ITA105_Lab_4_Match_comments.csv')
    df = df.dropna()
    
    df = pd.get_dummies(df, columns=['team', 'author'], drop_first=True)
    
    df['tokens'] = df['comment_text'].apply(preprocess_text)
    df['cleaned_text'] = df['tokens'].apply(lambda x: ' '.join(x))
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
    print(tfidf_matrix.shape)
    
    w2v_model = Word2Vec(sentences=df['tokens'], vector_size=10, window=3, min_count=1, workers=4)
    if 'xuất' in w2v_model.wv.key_to_index:
        print(w2v_model.wv.most_similar('xuất', topn=3))
        
    print("TF-IDF: Tạo vector thưa, coi mỗi từ là độc lập.")
    print("Word2Vec: Tạo vector dày, hiểu được ngữ cảnh từ vựng.")

def bai3_player_feedback():
    print("\n--- BÀI 3: FEEDBACK NGƯỜI CHƠI ---")
    df = pd.read_csv('ITA105_Lab_4_Player_feedback.csv')
    df = df.dropna()
    
    le = LabelEncoder()
    df['device_encoded'] = le.fit_transform(df['device'])
    df['player_type_encoded'] = le.fit_transform(df['player_type'])
    
    df['tokens'] = df['feedback_text'].apply(preprocess_text)
    df['cleaned_text'] = df['tokens'].apply(lambda x: ' '.join(x))
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
    print(tfidf_matrix.shape)
    
    w2v_model = Word2Vec(sentences=df['tokens'], vector_size=10, window=3, min_count=1, workers=4)
    if 'đẹp' in w2v_model.wv.key_to_index:
        print(w2v_model.wv.most_similar('đẹp', topn=3))
        
    print("TF-IDF: Phù hợp phân loại Sentiment với Machine Learning truyền thống.")
    print("Word2Vec: Ưu việt hơn khi dùng Deep Learning vì hiểu yếu tố phủ định.")

def bai4_album_reviews():
    print("\n--- BÀI 4: REVIEW ALBUM ---")
    df = pd.read_csv('ITA105_Lab_4_Album_reviews.csv')
    df = df.dropna()
    
    df = pd.get_dummies(df, columns=['genre', 'platform'], drop_first=True)
    
    df['tokens'] = df['review_text'].apply(preprocess_text)
    df['cleaned_text'] = df['tokens'].apply(lambda x: ' '.join(x))
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
    print(tfidf_matrix.shape)
    
    w2v_model = Word2Vec(sentences=df['tokens'], vector_size=10, window=3, min_count=1, workers=4)
    if 'sáng' in w2v_model.wv.key_to_index:
        print(w2v_model.wv.most_similar('sáng', topn=3))

def main():
    bai1_hotel_reviews()
    bai2_match_comments()
    bai3_player_feedback()
    bai4_album_reviews()

if __name__ == "__main__":
    main()