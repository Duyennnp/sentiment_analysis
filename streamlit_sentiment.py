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
import plotly.express as px

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

# file sô lượng bình luận
danhgia_sp_tg = pd.read_csv('danhgia_sp_tg.csv')

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
    Hàm dự đoán cảm xúc cho một hoặc nhiều bình luận và thêm emoji biểu cảm.
    
    Parameters:
    - comments: Danh sách các bình luận hoặc DataFrame chứa cột bình luận.
    - model: Mô hình đã huấn luyện.
    - vectorizer: Vectorizer đã huấn luyện
    
    Returns:
    - DataFrame với cột bình luận, dự đoán và emoji tương ứng.
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
    predictions = model.predict(comments_vec)
    comments_df["Prediction"] = predictions

    # Thêm cột emoji dựa trên dự đoán
    emoji_mapping = {
        "tích cực": "😊",  # Mặt cười vui vẻ
        "trung tính": "😐",   # Mặt bình thường
        "tiêu cực": "😞"   # Mặt buồn
    }
    comments_df["Emoji"] = comments_df["Prediction"].map(emoji_mapping)

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


# Hàm để trực quan hóa số lượng bình luận theo ngày
def visualize_product_info(df, product_code):
    # Lọc dữ liệu theo mã sản phẩm
    product_data = df[df['ma_san_pham'] == product_code]

    # Kiểm tra nếu không có dữ liệu cho mã sản phẩm này
    if product_data.empty:
        print(f"No data found for product code: {product_code}")
        return

    # Tính toán số lượng bình luận theo tháng và số sao trung bình theo tháng
    comment_count = product_data.groupby('thang_binh_luan').size().reset_index(name='so_luong_binh_luan')
    average_stars = product_data.groupby('thang_binh_luan')['so_sao'].mean().reset_index(name='so_sao_trung_binh')

    # Hợp nhất hai DataFrame
    merged_data = pd.merge(comment_count, average_stars, on='thang_binh_luan')

    # Vẽ biểu đồ tương tác bằng Plotly
    fig = px.bar(merged_data, x='thang_binh_luan', y='so_luong_binh_luan',
                 hover_data={'thang_binh_luan': '|%Y-%m', 'so_luong_binh_luan': True, 'so_sao_trung_binh': True},
                 labels={'so_luong_binh_luan': 'Số lượng bình luận', 'thang_binh_luan': 'Tháng bình luận'},
                 title=f'Số lượng bình luận và số sao theo tháng bình luận cho sản phẩm {product_code}')
    
    # Thêm đường biểu đồ cho số sao trung bình
    fig.add_scatter(x=merged_data['thang_binh_luan'].astype(str), y=merged_data['so_sao_trung_binh'], mode='lines+markers',
                    name='Số sao trung bình', yaxis='y2')

    # Tùy chỉnh trục y2 cho số sao trung bình
    fig.update_layout(
        yaxis2=dict(
            title='Số sao trung bình',
            overlaying='y',
            side='right'
        )
    )

    # Hiển thị biểu đồ
    st.plotly_chart(fig)




#---------------------------------------------------------------------------
# Thêm CSS tùy chỉnh
st.markdown(
    """
    <style>
    /* Nền cho toàn bộ ứng dụng */
    .stApp {
        background: linear-gradient(to top, #FFFFFF, #98FB98); /* Xanh lá đến trắng */
        color: black; /* Màu chữ trắng */
    }

    /* Tùy chỉnh tiêu đề */
    h1, h2, h3, h4, h5, h6 {
        color: black;
    }

    /* Tùy chỉnh phần sidebar */
    .css-1d391kg { /* Mã lớp cho sidebar */
        background: #333333; /* Màu nền sidebar */
        color: black; /* Màu chữ trong sidebar */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# GUI
# Tựa đề chính
st.title("🌟 Project 1: Sentiment Analysis")

# Menu với các mục và icon
st.sidebar.image("hasaki1.jpg")
menu = ["🏢 Business Objective", "📊 Build Project", "📈 New Prediction", "🛒 Product Analysis"]
choice = st.sidebar.selectbox('📂 Menu', menu)

# Sidebar thông tin
st.sidebar.markdown("""
    ### 🧑‍💻 Thành viên thực hiện:
    - Nguyễn Ngọc Phương Duyên
    - Mai Anh Sơn
""")
st.sidebar.markdown("""
    ### 📚 Giảng viên hướng dẫn:
    - Khuất Thùy Phương
""")
st.sidebar.write("### 🏫 Mã lớp: DL07_299T27_ON")
st.sidebar.write("### 🗓️ Thời gian thực hiện: 12/2024")


# Nội dung từng mục
if choice == '🏢 Business Objective':
    st.subheader("🏢 Business Objective")
    st.image("hasaki2.jpg")
    st.markdown("""
    ### Tổng quan về HASAKI:
    - HASAKI.VN là hệ thống cửa hàng mỹ phẩm chính hãng và dịch vụ chăm sóc sắc đẹp chuyên sâu với hệ thống cửa hàng trải dài trên toàn quốc.
    - Khách hàng có thể lên đây để lựa chọn sản phẩm, xem các đánh giá/ nhận xét cũng như đặt mua sản phẩm.
    """)
    st.write("#### 📌 Mục tiêu:")
    st.write("#### 1️⃣ **Xây dựng mô hình dự đoán phản hồi của khách hàng về sản phẩm**")
    st.image("sentiment.jpg", caption="Phân tích phản hồi khách hàng")
    st.write("#### 2️⃣ **Thực hiện phân tích sản phẩm**")
    st.image("product_analysis.jpg", caption="Phân tích sản phẩm Hasaki")

elif choice == '📊 Build Project':
    st.subheader("Build Project")
    st.write("#### 1. Xem dữ liệu")
    st.dataframe(df.head(3))
    st.dataframe(df.tail(3))  # Replace with `df.head()` for real data
    
    st.write("#### 2. Visualize Sentiment: 📉")
    st.image("SoLuong_BinhLuan.jpg", caption="Số lượng bình luận theo phân loại")
    st.markdown("""
    - Dữ liệu bị mất cân bằng với lớp tích cực chiếm đa số.
    - ✅ Phương án khắc phục: Tiến hành train trên tập hiện tại và cải thiện nếu cần.
    """)
    st.image("Imbalance.jpg", caption="Các phương pháp khắc phục mất cân bằng dữ liệu")

    st.write("#### 3. Xây dựng mô hình Random Forest")
    st.write("Training time: ⏱️ 17.519s")
    st.write("#### 4. Đánh giá mô hình:📊")
    st.markdown("""
    **Các chỉ số mô hình:**
    - Accuracy: 🎯 0.980
    - Precision: ✅ 0.980
    - Recall: 🔁 0.980
    - F1 score: 📈 0.980
    """)
    st.image("classification_report_v0.jpg", caption="Classification Report")
    st.image("confusion_matrix_rfr_graph.jpg", caption="Confusion Matrix")
    
    st.write("#### 5. Summary: Mô hình Random Forest với accuracy cao 0.98 cho thấy mô hình nhận diện khá tốt cho các lớp.")

elif choice == '📈 New Prediction':
    st.image('hasaki_banner_1.jpg', use_container_width=True)
    st.write("### Dự đoán cảm xúc bình luận khách hàng")
    st.markdown("""
    - ⭐ **<3 sao**: Tiêu cực
    - ⭐ **=3 sao**: Trung tính
    - ⭐ **>3 sao**: Tích cực
    """)
    st.write("#### Lựa chọn loại dữ liệu đưa vào:")
    flag = False
    lines = None
    type = st.radio("##### Lựa chọn loại dữ liệu:", options=("📂 Upload", "🖊️ Input"))
    if type == "📂 Upload":
        uploaded_file_1 = st.file_uploader("Chọn file:", type=['txt', 'csv'])
        if uploaded_file_1:
            st.write("✅ File uploaded thành công.")
            if uploaded_file_1 is not None:
                lines = pd.read_csv(uploaded_file_1, header=None, sep='\t')
                st.dataframe(lines)            
                comments = lines.iloc[:, 0].tolist()     
                flag = True                          
    elif type == "🖊️ Input":
        input_text = st.text_area("Nhập bình luận:")
        if st.button("Dự đoán 🔍"):
            st.write("✅ Kết quả dự đoán sẽ hiển thị ở đây.")
            if input_text !="":
                if input_text.strip():
                    comments = input_text.split("\n") #lines = [input_text .strip()]
                    flag = True
    
    # Hiển thị kết quả với emoji
    if flag:
        st.write("###### Content:")
        if len(comments) > 0:
            st.code(comments)
        
            # Ghi nhận thời gian bắt đầu
            start_time = time.time()
        
            # Thực hiện dự đoán
            results = predict_sentiment(comments, sentiment_model, vectorizer)
        
            # Ghi nhận thời gian kết thúc
            end_time = time.time()
        
            # Tính thời gian dự đoán
            prediction_time = end_time - start_time
        
             # Hiển thị thời gian dự đoán
            st.write(f"**Thời gian dự đoán**: {prediction_time:.3f} giây")
        
            # Hiển thị kết quả với emoji trên giao diện
            for _, row in results.iterrows():
                comment = row["Comment"]
                prediction = row["Prediction"]
                emoji = row["Emoji"]
                st.write(f"**Bình luận**: {comment}")
                st.write(f"**Dự đoán**: {prediction} {emoji}")
                st.markdown("---")      
           
    
elif choice == '🛒 Product Analysis':
    st.image('product.jpg', use_container_width=True)
    st.subheader("🛒 Product Analysis")
    
    # Tạo một danh sách mã sản phẩm và tên sản phẩm từ DataFrame 
    product_options = [f"{row['ma_san_pham']} - {row['ten_san_pham']}" for index, row in filtered_data.iterrows()]
    
    # Cho phép người dùng chọn hoặc nhập mã sản phẩm 
    selected_product_option = st.selectbox("🔍 Chọn sản phẩm:", product_options)

    # Lấy mã sản phẩm từ tùy chọn đã chọn 
    selected_product_id = int(selected_product_option.split(' - ')[0]) 
    st.write("##### Bạn đã chọn:", selected_product_option)

    # Hiển thị báo cáo cho mã sản phẩm đã chọn 
    if st.button("📊 Hiển thị báo cáo"):
        st.write(f"✅ Báo cáo sản phẩm {selected_product_option} sẽ hiển thị tại đây.")
        report_df = generate_product_report(selected_product_id, sp_analysis) 
        
        # Hiển thị báo cáo mà không sử dụng DataFrame
        st.write("### 🔍 List Positive:")
        for index, row in report_df.iterrows():
            st.write(f"- **Sản phẩm**: {row['ten_san_pham']}")
            st.write(f"  - **Số lượng tích cực**: {row['positive_count']}")
            st.write(f"  - **Danh sách từ khóa tích cực**: {row['positive_word_list']}")
            st.markdown("---")
        
        st.write("### 🔴 List Negative:")
        for index, row in report_df.iterrows():
            st.write(f"- **Sản phẩm**: {row['ten_san_pham']}")
            st.write(f"  - **Số lượng tiêu cực**: {row['negative_count']}")
            st.write(f"  - **Danh sách từ khóa tiêu cực**: {row['negative_word_list']}")
            st.markdown("---")
        
        SL_BinhLuan = visualize_product_info(danhgia_sp_tg, selected_product_id)