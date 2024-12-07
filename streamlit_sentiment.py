import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from underthesea import word_tokenize, pos_tag, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import regex
import string
import time
import joblib
from wordcloud import WordCloud, STOPWORDS
import ast

# 1. Read data 
df = pd.read_csv("processed_content.csv")
df = df.dropna()

# file phân tích sản phẩm
sp_analysis = pd.read_csv("sp_analysis.csv")

STOP_WORD_FILE = 'vietnamese-stopwords-D.txt'
with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    stop_words = file.read()

stop_words = stop_words.split('\n')

# chuyển list thành chuỗi
def convert_to_list(word_list_str):
    try:
        return ast.literal_eval(word_list_str)  # Chuyển đổi chuỗi JSON-like thành danh sách
    except (ValueError, SyntaxError):
        return []


sp_analysis['positive_word_list'] = sp_analysis['positive_word_list'].apply(convert_to_list)
sp_analysis['negative_word_list'] = sp_analysis['negative_word_list'].apply(convert_to_list)

sp_analysis['positive_words'] = sp_analysis['positive_word_list'].apply(lambda x: ' '.join(x))
sp_analysis['negative_words'] = sp_analysis['negative_word_list'].apply(lambda x: ' '.join(x))

# 2. Chia tập dữ liệu
X = df['processed_content']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Biến đổi văn bản thành đặc trưng (Bag-of-Words)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# 4. Load models 
# Đọc model Random Forest được chọn
# Tải mô hình bằng joblib
sentiment_model = joblib.load('model.pkl')


# 5. function cần thiết
# Dự đoán cảm đánh giá cho bình luận
def predict_sentiment(comments, model, vectorizer):
    """
    Hàm dự đoán cảm xúc cho một hoặc nhiều bình luận.
    
    Parameters:
    - comments: Danh sách các bình luận hoặc DataFrame chứa cột bình luận.
    - model: Mô hình đã huấn luyện.
    - vectorizer: Vectorizer đã huấn luyện
    
    Returns:
    - DataFrame với cột bình luận và dự đoán tương ứng.
    """
    # Nếu input là danh sách (một hoặc nhiều bình luận)
    if isinstance(comments, list):
        comments_df = pd.DataFrame(comments, columns=["Comment"])
    elif isinstance(comments, pd.DataFrame):
        comments_df = comments.rename(columns={comments.columns[0]: "Comment"})
    else:
        raise ValueError("Input phải là danh sách các bình luận hoặc DataFrame.")

    # Chuyển đổi văn bản sang vector
    comments_vec = vectorizer.transform(comments_df["Comment"])

    # Dự đoán cảm xúc
    comments_df["Prediction"] = model.predict(comments_vec)

    return comments_df

# Hàm để cung cấp thông tin liên quan sản phẩm
def generate_product_report(ma_san_pham, data):
    # Lấy thông tin sản phẩm cụ thể
    product_info = data[data['ma_san_pham'] == ma_san_pham]
    
    if product_info.empty:
        st.write(f"Sản phẩm với mã sản phẩm {ma_san_pham} không tồn tại.")
        return
    
    ten_san_pham = product_info['ten_san_pham'].values[0]
    positive_count = product_info['positive_count'].values[0]
    negative_count = product_info['negative_count'].values[0]
    positive_word_list = product_info['positive_word_list'].values[0]
    negative_word_list = product_info['negative_word_list'].values[0]
    positive_words = product_info['positive_words'].values[0]
    negative_words = product_info['negative_words'].values[0]
    processed_content = product_info['processed_content'].values[0]

    # Wordcloud cho nhận xét tích cực
    fig, ax = plt.subplots(figsize=(10, 5))
    positive_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(positive_words)
    ax.imshow(positive_wordcloud, interpolation='bilinear')
    ax.axis('off') 
    ax.set_title(f"Wordcloud for Positive Comments - mã sản phẩm {ma_san_pham}")
    st.pyplot(fig)
    plt.close(fig)

    # Wordcloud cho nhận xét tiêu cực
    fig, ax = plt.subplots(figsize=(10, 5))
    negative_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(negative_words)
    ax.imshow(negative_wordcloud, interpolation='bilinear')
    ax.axis('off') 
    ax.set_title(f"Wordcloud for Negative Comments - mã sản phẩm {ma_san_pham}")
    st.pyplot(fig)
    plt.close(fig)

    # Wordcloud cho nội dung bình luận (nằm ở giữa hàng dưới)
    fig, ax = plt.subplots(figsize=(10, 5))
    content_wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(processed_content)
    ax.imshow(content_wordcloud, interpolation='bilinear')
    ax.axis('off') 
    ax.set_title(f"Wordcloud cho mã sản phẩm {ma_san_pham}")
    st.pyplot(fig)
    plt.close(fig)
    
    # Tạo DataFrame chứa thông tin sản phẩm
    report_df = pd.DataFrame({
        'ma_san_pham': [ma_san_pham],
        'ten_san_pham': [ten_san_pham],
        'positive_count': [positive_count],
        'negative_count': [negative_count],
        'positive_word_list': [positive_word_list],
        'negative_word_list': [negative_word_list],
        'processed_content': [processed_content]
    })
    
    return report_df

# Tạo danh sách sản phẩm sẽ đưa vào selectbox
product_ids = [422217292, 422216990, 422220606, 422220469, 422222535, 422219399, 
                   100190059, 100150058, 100240016, 100230059, 100230064, 100230057, 100220035]

filtered_data = sp_analysis[sp_analysis['ma_san_pham'].isin(product_ids)][['ma_san_pham', 'ten_san_pham']]


#---------------------------------------------------------------------------
# GUI
st.title("Project 1")
st.write("## Sentiment Analysis")

menu = ["Business Objective", "Build Project", "New Prediction", "Product Analysis"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.markdown("""
                    #### Thành viên thực hiện:
                    - Nguyễn Ngọc Phương Duyên
                    - Mai Anh Sơn
                    """)
st.sidebar.markdown("""
                    #### Giảng viên hướng dẫn:
                    - Khuất Thùy Phương
                    """)
st.sidebar.write("""#### Mã lớp: DL07_299T27_ON""")
st.sidebar.write("""#### Thời gian thực hiện: 12/2024""")
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.markdown("""
                #### Tổng quan về HASAKI:
             - HASAKI.VN là hệ thống cửa hàng mỹ phẩm chính hãng và dịch vụ chăm sóc sắc đẹp chuyên sâu với hệ thống cửa hàng trải dài trên
             toàn quốc; và hiện đang là đối tác phân phối chiến lược tại thị trường Việt Nam của hàng loạt thương hiệu lớn
             - Khách hàng có thể lên đây để lựa chọn sản phẩm, xem các đánh giá/ nhận xét cũng như đặt mua sản phẩm.""")
    st.write("""#### 1. Xây dựng mô hình dự đoán phản hồi của khách hàng về sản phẩm""")
    st.write("""Xây dựng mô hình dự đoán giúp Hasaki.vn và các công ty đối tác có thể biết được 
             những phản hồi nhanh chóng của khách hàng về sản phẩm hay dịch vụ (tích cực, tiêu cực hay trung tính), 
             điều này giúp họ cải thiện sản phẩm/ dịch vụ → làm hài lòng khách hàng.""")  
    st.image("sentiment.jpg")
    st.write("""#### 2. Thực hiện phân tích sản phẩm""")
    st.write("""Khi chọn một sản phẩm cụ thể sẽ có những phân tích liên quan về sản phẩm như:
              mã sản phẩm, tên sản phẩm, số nhận xét tích cực và tiêu cực kèm wordcloud của từng loại, các keyword chính liên quan,... 
             để đối tác bán hàng nắm được tình hình sản phẩm và từ đó có những thay đổi tích cực.""")
    st.image("product_analysis.jpg")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("#### 1. Some data")
    st.dataframe(df.head(3))
    st.dataframe(df.tail(3))

    st.write("#### 2. Visualize Sentiment")
    st.image("SoLuong_BinhLuan.jpg") 
    st.write("""- Sơ đồ cho thấy dữ liệu bị mất cân bằng với lớp tích cực đang chiếm đa số, 
                do Hasaki là một hệ thống lớn và các đối tác cung cấp sản phẩm cho Hasaki đa số đều là các nhãn hàng đáng tin cậy, 
             nên sản phẩm và dịch vụ luôn được đánh giá cao.""")
    st.write("""##### Các phương pháp khắc phục:""")
    st.image("Imbalance.jpg")
    st.write("""- Sau khi xem xét không thực hiện các biện pháp trên mà vẫn thực hiện train mô hình trên tập dữ liệu hiện tại, 
             nếu kết quả đánh giá mô hình không khả quan sẽ thực hiện phương án cải thiện.""")

    st.write("#### 3. Build model Random Forest")
    st.write("Training time: 17.519s")
    
    st.write("#### 4. Evaluation")
    st.markdown("""
                **Các chỉ số mô hình:**
                - Accuracy: 0.980
                - Precision: 0.980
                - Recall: 0.980
                - F1 score: 0.980""")
    st.write("##### Classification report:")
    st.image("classification_report_v0.jpg")
    st.write("##### Confusion matrix:")
    st.image("confusion_matrix_rfr.jpg")
    st.image("confusion_matrix_rfr_graph.jpg")
    
    st.write("#### 5. Summary: Mô hình Random Forest với accuracy cao 0.98 cho thấy mô hình nhận diện khá tốt cho các lớp.")

elif choice == 'New Prediction':
    st.image('hasaki_banner_1.jpg', use_column_width=True)
    st.write("### Dự đoán cảm xúc bình luận của khách hàng Hasaki")
    st.write("#### Dựa vào 03 số sao đánh giá:")
    st.markdown("- **Nhỏ hơn 3 sao: Tiêu cực**")
    st.markdown("- **Bằng 3 sao: Trung tính**")
    st.markdown("- **Lớn hơn 3 sao: Tích cực**")
    st.write("#### Lựa chọn loại dữ liệu đưa vào:")
    flag = False
    lines = None
    type = st.radio("##### Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("##### Chọn file để upload với định dạng txt hoặc csv:", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None, sep='\t')
            st.dataframe(lines)            
            comments = lines.iloc[:, 0].tolist()     
            flag = True                          
    if type=="Input":        
        input_text  = st.text_area(label="##### Nhập một hoặc nhiều bình luận (mỗi bình luận trên một dòng):")
        if st.button("Dự đoán"):
            if input_text !="":
                if input_text.strip():
                    comments = input_text.split("\n") #lines = [input_text .strip()]
                    flag = True
    
    if flag:
        st.write("###### Content:")
        if len(comments)>0:
            st.code(comments)
            results = predict_sentiment(comments, sentiment_model, vectorizer)
            st.write(results)        
           
    
elif choice == 'Product Analysis':
    st.image('product.jpg', use_column_width=True)
    st.subheader("Product Report Generator")
    
    # Tạo một danh sách mã sản phẩm và tên sản phẩm từ DataFrame 
    product_options = [f"{row['ma_san_pham']} - {row['ten_san_pham']}" for index, row in filtered_data.iterrows()]
    
    # Cho phép người dùng chọn hoặc nhập mã sản phẩm 
    selected_product_option = st.selectbox("Chọn mã sản phẩm", product_options)

    # Lấy mã sản phẩm từ tùy chọn đã chọn 
    selected_product_id = int(selected_product_option.split(' - ')[0]) 
    st.write("##### Bạn đã chọn:", selected_product_option)

    # Hiển thị báo cáo cho mã sản phẩm đã chọn 
    if st.button("##### Hiển thị báo cáo"): 
        report_df = generate_product_report(selected_product_id, sp_analysis) 
        st.write("##### List Positive là:", report_df[['ten_san_pham', 'positive_count', 'positive_word_list']])
        st.write("##### List Negative là:", report_df[['ten_san_pham', 'negative_count', 'negative_word_list']])
        st.write("##### Bình luận sản phẩm:", report_df['processed_content'])
