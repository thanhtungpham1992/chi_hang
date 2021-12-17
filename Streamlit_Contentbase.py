# from numpy.lib.npyio import load
import streamlit as st 
import pandas as pd
import streamlit.components.v1 as stc
import webbrowser
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


## data
link_source = 'https://drive.google.com/uc?export=download&id='

link_product = link_source + '1FXITqJQrcnzdz1672YR5eBaheaEa2SuC' #https://drive.google.com/file/d/1QGEVPuV34xIfZMadexbnu3u1o4L_heYz/view?usp=sharing
#--------------
# Load data
@st.cache
def load_products():
    return pd.read_csv(link_product)


## stopwords file
# alphabet_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# link_s = 'https://drive.google.com/uc?export=download&id='

# link_sw = link_s + '1sf9lB6lX4MSluPM59zaqmCAfCyF1X1QH' #https://drive.google.com/file/d/1QGEVPuV34xIfZMadexbnu3u1o4L_heYz/view?usp=sharing
#--------------
# # Load data
# @st.cache
# def load_sw():
#     with open(link_sw, 'r', encoding = 'utf-8') as file:
#         stop_words=file.read()
#         stop_words = stop_words.split('\n')
#         stop_words.extend(alphabet_list)
#     return stop_words


# @st.cache
# def load_data():
#     return pd.read_csv('products.csv', index_col='Unnamed: 0')
# data = load_data()
# # print(data.info())
# # Data preprocessing
alphabet_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
STOP_WORD_FILE = 'vietnamese-stopwords.txt'
with open(STOP_WORD_FILE, 'r', encoding = 'utf-8') as file:
    stop_words = file.read()
    stop_words = stop_words.split('\n')
    stop_words.extend(alphabet_list)
stop_words.extend(alphabet_list)
# function xử lý chuỗi kỹ tự remove các ký tự đặc biệt
def remove_char(text):
    text = str(text)
    text = re.sub("[-.,:/\n()-+]+?|[0-9]", ' ', text.lower(), flags=re.MULTILINE).strip()
    text = re.sub("[…]|[•]|[–]|[*]|[≥]|[±]|[[]|[]]|[|]|[_]|[%]|[“]|[]|[<]", ' ', text.lower(), flags=re.MULTILINE)
    text = re.sub("\s\s+",' ',text).title()
    return(text)

# TFIDF and Cosine similarity
@st.cache
def tfidf_cosim(data):
    tf = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = stop_words)
    tfidf_matrix = tf.fit_transform(data['bag_words_wt'].apply(remove_char))
    # cosine similarity
    cosim = cosine_similarity(tfidf_matrix,tfidf_matrix)
    return cosim

# print(data.info())
# print(data[data['product_name']=='Tai nghe bluetooth không dây F9 True wireless Dock Sạc có Led Báo Pin Kép']['indice'].values[0])
def sort_rating(product_name, df, cosim):
    product_id = df[df['product_name']==product_name]['indice'].values[0]
    rating = list(enumerate(cosim[product_id]))
    sorted_rating = sorted(rating, key=lambda x: x[1], reverse=True)
    return sorted_rating

def top_product(products, ratings, brands, price_list, urls, images, num_of_rec):
    products = products[:num_of_rec]
    ratings = ratings[:num_of_rec]
    brands = brands[:num_of_rec]
    price_list = price_list[:num_of_rec]
    urls = urls[:num_of_rec]
    images = images[:num_of_rec]
    recommend_dict = {'product_name': products, 'rating': ratings, 'brand': brands, 'price':price_list, 'url': urls, 'image':images}
    recommend_df=pd.DataFrame(recommend_dict)
    return recommend_df

def main():
    st.markdown("<h1 style='font-family:Time news ronman;color:red'> Đồ án tốt nghiệp Data Science - Khóa 271<h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-family:Time news ronman; color:darkblue;backgroud-color:lightblue'><b>Content-Base Recommendation System<b></h2>", unsafe_allow_html=True)
    menu = ['Business Objective', 'Build Project', 'Recommendation']
    choice = st.sidebar.selectbox('Menu', menu)
    if choice == 'Business Objective':
        st.markdown("<h3 style='font-family:Time news ronman; color:darkbrown;'><b>Tiki and Recommendation</b></h3>", unsafe_allow_html=True)
        st.markdown("<p style='font-family:Time news ronman;'><b>Tiki</b> Là công ty sáng lập vào tháng 3/2010, xuất phát điểm chỉ là website bán sách tiếng Anh trực tuyến được khơi nguồn cảm hứng từ niềm đam mê với sách và nhận thấy nhu cầu rộng lớn của người Việt với các loại sách ngoại ngữ\
Tháng 3/2012 khi nhận được vốn đầu tư của ông Soichi Tajima Chủ tịch kiêm CEO của Quỹ đầu tư CyberAgent Ventures Inc. Với khả năng khai thác tốn khoản đầu tư lớn này, ông Sơn đã phát triển được mạng lưới khách hàng rộng lớn hớn. Tái cấu trúc lại hệ thống kho bãi và nhà cung cấp hàng hóa.\
Vào năm 2016, công ty cổ phần Tiki được đầu tư số tiền 384 tỷ đồng tương đương 38% cổ phần từ Công ty cổ phần VNG. Năm 2018, Tiki tiếp tục được đầu tư từ JDar Inc công ty bán lẻ lớn nhất của Trung Quốc với số vốn 1000 tỷ đồng. Tiki đã sử dụng nguồn vốn đầu tư này để nâng cấp hệ thống kho bãi, phát triển phần mềm và hệ thống thanh toán trực tuyến và phát triển thị trường.\
<br><b>SẢN PHẨM TRÊN TIKI</b><br>Các ngành đang được kinh doanh nổi bật trên trang thương mại điện tử Tiki.vn có thể kể đến gồm Điện Thoại- Máy Tính Bảng, Thiết Bị Số -Phụ Kiện Số, Điện Gia Dụng, Sách, Nhà Cửa Đời Sống, Làm Đẹp - Sức Khỏe, Mẹ & Bé, ...\
<br><b>MÔ HÌNH KINH DOANH B2C:</b><br>Nghĩa là Tiki là nơi kết nối, tạo không gian giao dịch giữa doanh nghiệp và người tiêu dùng, nhằm mục đích mang đến cho khách hàng sự tiện lợi, dễ dàng khi mua sắm online.\
Recommendation system có thể giúp quảng bá sản phẩm mới, kiểm soát hàng tồn kho của riêng, làm nổi bật các mặt hàng có giá khuyến mại, khi thanh lý hoặc được cung cấp quá nhiều. Nó cung cấp sự linh hoạt để điều chỉnh chính xác những mục nào được hệ thống đề xuất đánh dấu tới khách hàng.</p>", unsafe_allow_html=True)
    elif choice == 'Build Project':
        st.subheader("Data Understanding")
        st.markdown("<b>Dataframe</b>", unsafe_allow_html=True)
        st.image('data_preprocessing.png')
        st.subheader("Data Cleaning and Preparing")
        st.subheader("Some Insights")
        
        st.write("Dữ liệu thiếu sau khi xử lý")
        st.image('missing_value.png')
        colb, colc = st.beta_columns([1,1])
        with colb:
            st.write("Tên sản phẩm")
            st.image('product_name.png')

        with colc:
            st.write("Đánh giá trung bình")
            st.image('product_rating.png')                
        cole, colf = st.beta_columns([1,1])
        with cole:
            st.write("Group of item")
            st.image('group.png')

        with colf:
            st.write("category of item")
            st.image('category.png')
    else: 
        st.subheader("Recommend Base on product name")
        # input = st.text_input("Nhập chính xác tên sản phẩm bạn muốn tìm kiếm")
        # st.dataframe(data)
        with st.form(key='Tìm kiếm sản phẩm tương tự'):
            input = st.selectbox('Chọn tên sản phẩm bạn muốn tìm kiếm:', data['product_name'])
            submit_button = st.form_submit_button(label='Recommendation')
            num_of_rec = st.sidebar.number_input("Number items", 1, 10, 1 )
        # cosim = load('cosim.npy')
        if submit_button:
            if input is not None:
                sorted_rating=sort_rating(input, data, cosim)
                products = [data[products[0]==data['indice']]['product_name'].values[0] for products in sorted_rating]
                ratings = [data[ratings[0]==data['indice']]['pro_rating'].values[0] for ratings in sorted_rating]
                brands = [data[brands[0]==data['indice']]['brand'].values[0] for brands in sorted_rating]
                price_list = [data[price_list[0]==data['indice']]['price'].values[0] for price_list in sorted_rating]
                urls = [data[urls[0]==data['indice']]['url'].values[0] for urls in sorted_rating]
                images = [data[images[0]==data['indice']]['image'].values[0] for images in sorted_rating]
                result = top_product(products, ratings, brands, price_list, urls, images, num_of_rec)
                print(result)
                st.write(result)
                def product_in_col(n,prd_to_show):   
                    a = st.image(prd_to_show['image'][n], use_column_width=True)
                    e = st.write('Product Name: ', prd_to_show['product_name'][n])
                    c = st.write('Product Brand: ',prd_to_show['brand'][n], use_column_width=True)
                    d = st.write('Product Rating: ', prd_to_show['rating'][n])
                    f = st.write('Product Price: ', prd_to_show['price'][n])
                    # b = st.write('Product Link: ', prd_to_show['url'][n])
                    if st.button('Open Product',prd_to_show['url'][n]):
                        webbrowser.open(prd_to_show['price'][n])
                    return a, e, c, d, f

                st.subheader('Sản phẩm gợi ý')
                col1, col2, col3 = st.beta_columns([1,1,1])

                with col1:
                    product_in_col(0,result)
                with col2:
                    product_in_col(1,result)
                with col3:
                    product_in_col(2,result)

                col4, col5, col6 = st.beta_columns([1,1,1])
                with col4:
                    product_in_col(3,result)
                with col5:
                    product_in_col(4,result)
                with col6:
                    product_in_col(5,result)
        # st.subheader("Recommend Base on product name and price")


if __name__=='__main__':
    # data = load_data()
    data = load_products()
    # stop_words = load_sw()
    cosim = tfidf_cosim(data)
    main()