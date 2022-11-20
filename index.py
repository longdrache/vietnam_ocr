from pickle import bytes_types
import streamlit as st
from PIL import Image
from predict import  predict
import io

st.markdown('<h1 style="text-align:center;padding:20px;">Nhận dạng chữ cái Tiếng Việt</h1>',unsafe_allow_html=True)
image = Image.open('header.png')
st.image(image)
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    image_pred = Image.open(io.BytesIO(bytes_data))
    st.image(image_pred)
   
    with st.spinner('Wait for it...'):
        st.markdown('<p style="text-align:center;padding:30px;font-size:30px">{}</p>'.format(predict(image_pred)), unsafe_allow_html=True)

    

