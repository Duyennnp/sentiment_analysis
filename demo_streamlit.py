import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Using menu
st.title("Trung Tâm Tin Học")
menu = ["Home", "Capstone Project", "Sử dụng các điều khiển", "Hiển thị chart"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':    
    st.subheader("[Trang chủ](https://csc.edu.vn)")  
elif choice == 'Capstone Project':    
    st.subheader("[Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/do-an-tot-nghiep-data-science_310)")
    st.write("""### Có 2 chủ đề trong khóa học:
    - Topic 1: Sentiment Analysis
    - Topic 2: Recommendation System
    - ...""")
elif choice == 'Sử dụng các điều khiển':
    # Sử dụng các điều khiển nhập
    # 1. Text
    st.subheader("1. Text")
    name = st.text_input("Enter your name")
    st.write("Your name is", name)
    # 2. Slider
    st.subheader("2. Slider")
    age = st.slider("How old are you?", 1, 100, 20)
    st.write("I'm", age, "years old.")
    # 3. Checkbox
    st.subheader("3. Checkbox")
    if st.checkbox("I agree"):
        st.write("Great!")
    # 4. Radio
    st.subheader("4. Radio")
    status = st.radio("What is your status?", ("Active", "Inactive"))
    st.write("You are", status)
    # 5. Selectbox
    st.subheader("5. Selectbox")
    occupation = st.selectbox("What is your occupation?", ["Student", "Teacher", "Others"])
    st.write("You are a", occupation)
    # 6. Multiselect
    st.subheader("6. Multiselect")
    location = st.multiselect("Where do you live?", ("Hanoi", "HCM", "Danang", "Hue"))
    st.write("You live in", location)
    # 7. File Uploader
    st.subheader("7. File Uploader")
    file = st.file_uploader("Upload your file", type=["csv", "txt"])
    if file is not None:
        st.write(file)    
    # 9. Date Input
    st.subheader("9. Date Input")
    date = st.date_input("Pick a date")
    st.write("You picked", date)
    # 10. Time Input
    st.subheader("10. Time Input")
    time = st.time_input("Pick a time")
    st.write("You picked", time)
    # 11. Display JSON
    st.subheader("11. Display JSON")
    json = st.text_input("Enter JSON", '{"name": "Alice", "age": 25}')
    st.write("You entered", json)
    # 12. Display Raw Code
    st.subheader("12. Display Raw Code")
    code = st.text_area("Enter code", "print('Hello, world!')")
    st.write("You entered", code)
    # Sử dụng điều khiển submit
    st.subheader("Submit")
    submitted = st.button("Submit")
    if submitted:
        st.write("You submitted the form.")
        # In các thông tin phía trên khi người dùng nhấn nút Submit
        st.write("Your name is", name)
        st.write("I'm", age, "years old.")
        st.write("You are", status)
        st.write("You are a", occupation)
        st.write("You live in", location)
        st.write("You picked", date)
        st.write("You picked", time)
        st.write("You entered", json)
        st.write("You entered", code)

elif choice == 'Hiển thị chart':
    # Hiển thị chart
    # 1. Line Chart
    st.subheader("1. Line Chart")
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [10, 20, 30, 40, 50]
    })
    st.line_chart(df)
    # 2. Area Chart
    st.subheader("2. Area Chart")
    st.area_chart(df)
    # 3. Bar Chart
    st.subheader("3. Bar Chart")
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [10, 20, 30, 40, 50]
    })
    # vẽ biểu đồ bằng matplotlib
    fig, ax = plt.subplots()
    ax.bar(df['x'], df['y'])
    st.pyplot(fig)    
    # 4. Plot Map
    st.subheader("4. Plot Map")
    df = pd.DataFrame({
        'lat': [21.03, 21.02, 21.01],
        'lon': [105.85, 105.86, 105.85],
        'name': ['Hàng Trống', 'Phan Chu Trinh', 'Lê Đại Hành']
    })
    st.map(df)
    # 5. Plot Data
    # Dùng thư viện seaborn
    st.subheader("5. Plot Data")
    # Tạo dataframe
    data = {
        'x': [1, 2, 3, 4, 5],
        'y': [10, 20, 30, 40, 50]
    }
    df = pd.DataFrame(data)
    # Vẽ biểu đồ
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x='x', y='y', ax=ax)
    st.pyplot(fig)
# Done
    
    
        

        
        

    



