import time
import torch
import faiss
import pathlib
from PIL import Image

import streamlit as st
from streamlit_cropper import st_cropper

from src.feature_extraction import MyVGG16, MyResnet50, RGBHistogram, LBP
from src.dataloader import get_transformation
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device('cpu')
image_root = './dataset/photos'
feature_root = './dataset/feature'


def get_image_list(image_root):
    image_root = pathlib.Path(image_root)
    image_list = list()
    for image_path in image_root.iterdir():
        if image_path.exists():
            image_list.append(image_path)
    image_list = sorted(image_list, key=lambda x: x.name)
    return image_list


def retrieve_image(img, feature_extractor):
    if (feature_extractor == 'VGG16'):
        extractor = MyVGG16(device)
    elif (feature_extractor == 'Resnet50'):
        extractor = MyResnet50(device)
    elif (feature_extractor == 'RGBHistogram'):
        extractor = RGBHistogram(device)
    elif (feature_extractor == 'LBP'):
        extractor = LBP(device)

    transform = get_transformation()

    img = img.convert('RGB')
    image_tensor = transform(img)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    feat = extractor.extract_features(image_tensor)

    indexer = faiss.read_index(feature_root + '/' + feature_extractor + '.index.bin')

    _, indices = indexer.search(feat, k=11)

    return indices[0]


def sort_images_by(criteria, indices, image_list):
    if criteria == "Khoảng cách đặc trưng gần nhất":
        sorted_indices = sorted(indices)  # sắp xếp tăng dần
    elif criteria == "Khoảng cách đặc trưng xa nhất":
        sorted_indices = sorted(indices, reverse=True) # Sắp xếp giảm dần
    else:
        sorted_indices = indices
    return sorted_indices



def main():
    st.set_page_config(page_title="Tìm kiếm Ảnh", layout="wide")
    st.markdown(
        """<style>/* CSS cho toàn bộ trang */ * { margin: 0; padding: 0; box-sizing: border-box; } .main-title { font-size: 36px; text-align: center; color: #2E8B57; margin-top: 20px; margin-bottom: 40px; } .header { font-size: 24px; color: #2E8B57; border-bottom: 2px solid #2E8B57; margin-bottom: 10px; padding-bottom: 8px; } .subheader { color: #FF6347; font-size: 20px; margin-top: 16px; margin-bottom: 10px; } .upload-area { border: 2px dashed #2E8B57; border-radius: 10px; background-color: #FFFFFF; padding: 20px; text-align: center; margin-bottom: 20px; } .stButton>button { background-color: #2E8B57; color: white; border-radius: 8px; font-size: 18px; padding: 10px 20px; transition: background-color 0.3s; } .stButton>button:hover { background-color: #228B22; } </style>""",
        unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">🖼️ TÌM KIẾM ẢNH DỰA TRÊN NỘI DUNG</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="header">🔍 TRUY VẤN</div>', unsafe_allow_html=True)
        st.markdown('<div class="subheader">Chọn bộ trích xuất đặc trưng</div>', unsafe_allow_html=True)
        option = st.selectbox('', ('Resnet50', 'VGG16', 'RGBHistogram', 'LBP'))

        st.markdown('<div class="subheader">Tải lên hình ảnh</div>', unsafe_allow_html=True)
        img_file = st.file_uploader('', type=['png', 'jpg'], help="Tải lên hình ảnh để bắt đầu tìm kiếm")

        if img_file:
            img = Image.open(img_file)
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')

            st.write("**Xem trước:**")
            preview_img = cropped_img.copy()
            preview_img.thumbnail((200, 200))
            st.image(preview_img, caption="Hình ảnh đã cắt", use_container_width=True)

    with col2:
        st.markdown('<div class="header">📋 KẾT QUẢ</div>', unsafe_allow_html=True)

        if img_file:
            loading_message = st.empty()
            loading_message.info('**Đang tìm kiếm... Vui lòng đợi.**')
            start = time.time()

            retriev = retrieve_image(cropped_img, option)
            image_list = get_image_list(image_root)
            end = time.time()
            loading_message.empty()

            st.success(f'**Hoàn thành sau {end - start:.2f} giây**')

            # Lựa chọn tiêu chí sắp xếp
            sort_criteria = st.selectbox('Chọn tiêu chí sắp xếp', ('Khoảng cách đặc trưng gần nhất', 'Khoảng cách đặc trưng xa nhất'))

            # Sắp xếp kết quả tìm kiếm
            sorted_indices = sort_images_by(sort_criteria, retriev, image_list)

            # Hiển thị kết quả
            st.markdown('<div class="subheader">Kết quả tìm thấy</div>', unsafe_allow_html=True)
            retrieved_cols = st.columns(2)
            for idx, col in enumerate(retrieved_cols):
                image_path = image_list[sorted_indices[idx]]
                image = Image.open(image_path)
                col.image(image, caption=f"Kết quả {idx + 1}", use_container_width=True)

            st.markdown("---")
            st.markdown('<div class="subheader">Kết quả khác</div>', unsafe_allow_html=True)
            more_cols = st.columns(3)
            for i, col in enumerate(more_cols):
                for u in range(i + 2, 11, 3):
                    image_path = image_list[sorted_indices[u]]
                    image = Image.open(image_path)
                    col.image(image, caption=f"Kết quả {u + 1}", use_container_width=True)


if __name__ == '__main__':
    main()
