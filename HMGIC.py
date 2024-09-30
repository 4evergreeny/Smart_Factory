
# 라이브러리 불러오기 
# import os
# import pandas as pd
# import numpy as np
# import random
# import time
# import datetime
# from PIL import Image, ImageOps,ImageDraw, ImageFont
# from datetime import timedelta
# import joblib
# from xgboost import XGBClassifier
# from keras.models import load_model
# from haversine import haversine
# from urllib.parse import quote
# import streamlit as st
# from streamlit_folium import st_folium
# import folium
# import branca
# from geopy.geocoders import Nominatim
# import ssl
# from urllib.request import urlopen
# import requests
# import csv

# from gtts import gTTS
# from playsound import playsound

# import plotly.express as px
# import altair as alt

# import streamlit as st
# from streamlit_folium import st_folium
# from folium.plugins import MarkerCluster
# from io import BytesIO
# import cv2
# import tensorflow as tf
# from datetime import datetime
# import sqlite3
# import io
# import base64

# from svglib.svglib import svg2rlg
# import hashlib

# import cv2
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
# import av

# from pymelsec import Type4E

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import io
import base64

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -------------------- ▼ 필요 함수 생성 코딩 ▼ --------------------
def load_image(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 컨텐츠 타입 확인
        content_type = response.headers.get('content-type')
        if 'image' not in content_type:
            st.error(f"URL이 이미지가 아닙니다. 컨텐츠 타입: {content_type}")
            return None
        
        # 이미지 크기 확인
        content_length = int(response.headers.get('content-length', 0))
        if content_length == 0:
            st.error("이미지 데이터가 비어있습니다.")
            return None
        
        image_data = BytesIO(response.content)
        
        # 이미지 포맷 확인
        try:
            img = Image.open(image_data)
            img.verify()  # 이미지 데이터 검증
            image_data.seek(0)  # BytesIO 객체 포인터를 처음으로 되돌림
            return Image.open(image_data)
        except Exception as e:
            st.error(f"이미지 형식이 올바르지 않습니다: {e}")
            return None
        
    except requests.exceptions.RequestException as e:
        st.error(f"이미지를 다운로드하는 데 실패했습니다: {e}")
    except Exception as e:
        st.error(f"알 수 없는 오류가 발생했습니다: {e}")
    
    return None

def load_image(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"이미지를 불러오는 데 실패했습니다: {e}")
        return None

# 웹캠 이미지 조정 함수
def adjust_frame(frame):
    # BGR에서 RGB로 변환 (OpenCV는 기본적으로 BGR 형식을 사용)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

def load_svg(url):
    response = requests.get(url)
    return response.content.decode('utf-8')

def render_svg(svg_content, height):
    b64 = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
    html = f'<img src="data:image/svg+xml;base64,{b64}" style="height:{height}px; vertical-align:middle;"/>'
    return html

# -------------------- ▼ 1-0그룹 Streamlit 로그인 화면 구성 Tab 생성 START ▼ --------------------

# 데이터베이스 연결
conn = sqlite3.connect('users.db')
c = conn.cursor()

# 사용자 테이블 생성
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY, password TEXT, role TEXT)''')

# 함수 정의
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# def load_svg(url):
#    response = requests.get(url)
#    return response.content.decode('utf-8')


#-------- 두 번째 카매라용 추가 함수 --------

def load_svg(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content.decode('utf-8')
    except requests.exceptions.RequestException as e:
        st.error(f"로고를 불러오는 데 실패했습니다: {e}")
        return ""

def get_webcam_frame(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error(f"웹캠 {camera_index}를 열 수 없습니다.")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        st.error(f"웹캠 {camera_index}에서 프레임을 읽을 수 없습니다.")
        return None
    return adjust_frame(frame)
#-----------

def render_svg(svg_content, height):
    b64 = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
    html = f'<img src="data:image/svg+xml;base64,{b64}" style="height:{height}px; vertical-align:middle;"/>'
    return html

def register_user():
    st.subheader("새 사용자 등록")
    new_user = st.text_input("사용자명")
    new_password = st.text_input("비밀번호", type="password")
    new_role = st.selectbox("권한", ["일반", "관리자"])
    
    if st.button("등록"):
        hashed_pw = hash_password(new_password)
        c.execute("INSERT INTO users VALUES (?, ?, ?)", (new_user, hashed_pw, new_role))
        conn.commit()
        st.success("등록 완료!")

def login_user():
    st.subheader("로그인")
    username = st.text_input("사용자명")
    password = st.text_input("비밀번호", type="password")
    
    if st.button("로그인"):
        hashed_pw = hash_password(password)
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_pw))
        result = c.fetchone()
        if result:
            st.success("로그인 성공!")
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.session_state['role'] = result[2]
            st.experimental_rerun()
        else:
            st.error("잘못된 사용자명 또는 비밀번호")

# -------------------- ▼ 1-0그룹 Streamlit 웹 화면 구성 Tab 생성 START ▼ --------------------
st.set_page_config(layout="wide")

# 로고 로딩을 전역 범위로 이동
logo_url = "https://www.hyundai.com/contents/images/logo.svg"
svg_content = load_svg(logo_url)

def main():
    # 제목의 폰트 크기를 가져옵니다 (기본값은 35px입니다)
    title_size = 35  # st.title의 기본 폰트 크기

    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            {render_svg(svg_content, title_size)}
            <h1 style="margin-left: 10px;">현대자동차 통합 MES</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        option = st.selectbox("선택", ["로그인", "등록"])
        if option == "로그인":
            login_user()
        else:
            register_user()
    else:
        st.write(f"환영합니다, {st.session_state['username']}님!")
        if st.session_state['role'] == "관리자":
            st.write("관리자 페이지입니다.")
        else:
            st.write("일반 사용자 페이지입니다.")

        if st.button("로그아웃"):
            st.session_state['logged_in'] = False
            st.experimental_rerun()
# 제목의 폰트 크기를 가져옵니다 (기본값은 35px입니다)
title_size = 35  # st.title의 기본 폰트 크기

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        {render_svg(svg_content, title_size)}
        <h1 style="margin-left: 10px;">현대자동차 통합 MES</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# 사용자 정의 객체 추가
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

# 커스텀 객체 등록
tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = FixedDepthwiseConv2D

# 모델 로드
model = tf.keras.models.load_model('color_model.h5')

# 라벨 로드
with open('color_labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    return predictions[0]

def draw_bounding_box(image, prediction):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # 가장 높은 확률의 클래스 선택
    class_id = np.argmax(prediction)
    score = prediction[class_id]
    label = labels[class_id]
    
    # 바운딩 박스 그리기 (예시로 전체 이미지의 80%를 차지하도록 설정)
    box_width = int(width * 0.8)
    box_height = int(height * 0.8)
    x = int((width - box_width) / 2)
    y = int((height - box_height) / 2)
    
    draw.rectangle([x, y, x + box_width, y + box_height], outline="red", width=2)
    
    # 라벨과 확률 표시
    font = ImageFont.load_default()
    text = f"{label}: {score:.2f}"
    draw.text((x, y - 20), text, fill="red", font=font)
    
    return image

st.empty()

# 데이터베이스 연결
conn = sqlite3.connect('paint_quality.db')
c = conn.cursor()

# 테이블 생성
c.execute('''CREATE TABLE IF NOT EXISTS inspections
             (timestamp TEXT, result TEXT)''')

# 공장 데이터
factories = {
    "울산공장": {
        "lat": 35.5384, 
        "lon": 129.3114, 
        "info": [
            "단일 자동차 공장 중 세계 최대 규모",
            "5개 독립 제조 공장, 엔진 및 트랜스미션 공장",
            "수출 부두, 품질 관리 센터"
        ],
        "image": "https://www.hyundai.com/content/dam/hyundai/ww/en/images/about-hyundai/corporate/networks/corp-manufacturing-ulsan-plant-thumb.jpg",
        "products": ["코나", "투싼", "싼타페", "팰리세이드"]
    },
    "아산공장": {
        "lat": 36.7851, 
        "lon": 126.9767, 
        "info": ["첨단 자립형 공장",
                 "수출용 승용차 생산: 쏘나타, 그랜저(Azera) 등",
                 "친환경 태양열 루프탑 농장 운영"
                ],
        "image": "https://www.hyundai.com/content/dam/hyundai/ww/en/images/about-hyundai/corporate/networks/corp-manufacturing-asan-plant-thumb.jpg",
        "products": ["소나타", "그랜저"]
    },
    "전주공장": {
        "lat": 35.8468, 
        "lon": 127.1229, 
        "info": ["글로벌 상용차 제조 기지",
                 "세계 최대의 상용차 생산 공장",
                 "세계 최초 연료 전지 전기 트럭 제조"
                ],
        "image": "https://www.hyundai.com/content/dam/hyundai/ww/en/images/about-hyundai/corporate/networks/corp-manufacturing-jeonju-plant-thumb.jpg",
        "products": ["트럭", "버스"]
    }
}

def preprocess_image(frame):
    resized = cv2.resize(frame, (224, 224))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

def preprocess_image(image):
    # PIL Image를 numpy 배열로 변환
    img_array = np.array(image)
    # 이미지 크기 조정
    resized = cv2.resize(img_array, (224, 224))
    # 정규화
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

def predict_quality(image):
    if model is None:
        return "모델 로딩 실패", 0
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)
    class_names = ['Rock', 'Paper', 'Scissors', 'Fxxk', 'Promise']  # 실제 클래스 이름으로 변경해야 합니다
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

def add_border(image, border_size=1, border_color='gray'):
    return ImageOps.expand(image, border=border_size, fill=border_color)

st.empty() 

#
# 데이터베이스 연결 및 테이블 생성 함수
def create_connection():
    conn = sqlite3.connect('classification_results.db')
    return conn

def create_table(conn, labels):
    c = conn.cursor()
    columns = ", ".join([f"{label} REAL" for label in labels])
    c.execute(f'''CREATE TABLE IF NOT EXISTS results
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  production_number TEXT,
                  capture_time DATETIME,
                  {columns})''')
    conn.commit()

# 생산번호 생성 함수
def generate_production_number():
    return datetime.now().strftime("%Y%m%d%H%M%S")

# 촬영시간 생성 함수
def get_capture_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 결과를 데이터베이스에 저장하는 함수
def save_result(conn, predictions, labels):
    c = conn.cursor()
    production_number = generate_production_number()
    capture_time = get_capture_time()
    values = [production_number, capture_time] + list(predictions)
    placeholders = ", ".join(["?" for _ in range(len(values))])
    columns = "production_number, capture_time, " + ", ".join(labels)
    c.execute(f"INSERT INTO results ({columns}) VALUES ({placeholders})", values)
    conn.commit()

# 결과를 표 형태로 보여주는 함수
def show_results_table(conn):
    df = pd.read_sql_query("SELECT * FROM results ORDER BY capture_time DESC", conn)
    st.table(df)

def main():
    col1, col2 = st.columns([3, 2])
    with col1:
        # 지도 생성
        m = folium.Map(location=[36.5, 127.5], zoom_start=7, min_zoom=7, max_zoom=12)

        # 공장 마커 추가 (팝업 내용 수정)
        for name, data in factories.items():
            popup_content = f"""
            <div style="width:300px">
                <h4>{name}</h4>
                <img src="{data['image']}" width="100%" alt="{name}">
                <p>{data['info'][0]}<br>{data['info'][1]}<br>{data['info'][2]}</p>
            </div>
            """
            folium.Marker(
                [data['lat'], data['lon']],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=name
            ).add_to(m)

        # Streamlit에 지도 표시 (크기 증가)
        st_data = st_folium(m, width=800, height=600)

    with col2:
        # 공장 선택 셀렉트 박스 (고유 키 추가)
        selected_factory = st.selectbox("공장 선택", list(factories.keys()), key="factory_select")
    st.empty()
    
    # 공장 정보 탭 (지도 아래에 배치)
    if selected_factory:
        factory_data = factories[selected_factory]

        # 탭 생성
        tab1, tab2 = st.tabs(["공장 정보", "생산 정보"])

        with tab1:
            st.subheader(f"{selected_factory} 정보")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(factory_data['image'], use_column_width=True)
            
            with col2:
                st.write(f"위치: 위도 {factory_data['lat']}, 경도 {factory_data['lon']}")
                st.write("설명:")
                for info in factory_data['info']:
                    st.write(f"- {info}")

        with tab2:
            st.subheader(f"{selected_factory} 생산 차종")
            for product in factory_data['products']:
                st.write(f"- {product}")

    st.header("생산 차량 실시간 이미지 분류")

    # 라이브 이미지와 평가 후 이미지를 2열로 표시
    col1, col2 = st.columns(2)

    with col1:
        # Streamlit의 카메라 입력 사용
        camera_image = st.camera_input("머신러닝 이미지 분류", key="camera_input")

    with col2:
        st.write("이미지를 촬영하면 아래에 분류 결과가 표시됩니다.")

        if camera_image is not None:
            # 이미지를 numpy array로 변환
            image = Image.open(camera_image).convert('RGB')
            frame = np.array(image)

             # 이미지에 테두리 추가
            bordered_image = add_border(image)

            try:
                # 예측 수행
                prediction = predict_image(image)
                
                # 바운딩 박스 그리기
                result_image = draw_bounding_box(image.copy(), prediction)
                st.image(result_image, caption="분류 결과", use_column_width=True)
                
                # 모든 클래스의 확률 표시
                st.write("클래스별 확률:")
                for i, prob in enumerate(prediction):
                    st.write(f"{labels[i]}: {prob:.2f}")

            except Exception as e:
                st.error(f"예측 중 오류 발생: {e}")

    # # 두 번째 웹캠 (새로 추가)
    # st.header("작업자 실시간 모니터링 시스템")
    # col3, col4 = st.columns(2)

    # with col3:
    #     ctx = webrtc_streamer(key="camera2", video_processor_factory=VideoProcessor)
    #     if ctx.video_transformer:
    #         snapshot = ctx.video_transformer.snapshot
    #         if snapshot is not None:
    #             st.image(snapshot, channels="BGR", caption="작업자 실시간 모니터링 화면")

    # with col4:
    #     st.write("작업자가 접근하면 공정이 멈출 수 있습니다.")
    #     if ctx.video_transformer:
    #         snapshot = ctx.video_transformer.snapshot
    #         if snapshot is not None:
    #             try:
    #                 # 이미지를 PIL Image로 변환
    #                 image2 = Image.fromarray(cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB))
                    
    #                 # 예측 수행
    #                 prediction2 = predict_image(image2)

    #                 # 바운딩 박스 그리기
    #                 result_image2 = draw_bounding_box(image2.copy(), prediction2)
    #                 st.image(result_image2, caption="분류 결과 (카메라 2)", use_column_width=True)
                    
    #                 # 모든 클래스의 확률 표시
    #                 st.write("클래스별 확률 (카메라 2):")
    #                 for i, prob in enumerate(prediction2):
    #                     st.write(f"{labels[i]}: {prob:.2f}")

    #             except Exception as e:
    #                 st.error(f"예측 중 오류 발생 (카메라 2): {e}")


        if 'logged_in' in st.session_state and st.session_state['logged_in']:
            st.write(f"환영합니다, {st.session_state['username']}님!")
            
            # 기존의 col1, col2는 그대로 두고, 새로운 col3, col4를 생성합니다.
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("첫 번째 웹캠")
                webcam_placeholder1 = st.empty()
            
            with col4:
                st.subheader("두 번째 웹캠")
                webcam_placeholder2 = st.empty()
            
            while True:
                webcam_image1 = get_webcam_frame(0)
                if webcam_image1 is not None:
                    webcam_placeholder1.image(webcam_image1, channels="RGB")
                
                webcam_image2 = get_webcam_frame(1)
                if webcam_image2 is not None:
                    webcam_placeholder2.image(webcam_image2, channels="RGB")
                
                time.sleep(0.1)  # 프레임 갱신 간격 (초)

    
        # if camera_image2 is not None:
        #     # 이미지를 numpy array로 변환
        #     image2 = Image.open(camera_image2).convert('RGB')
        #     frame2 = np.array(image2)

        #     # 이미지에 테두리 추가
        #     bordered_image2 = add_border(image2)

            # try:
            #     # 예측 수행
            #     prediction2 = predict_image(image2)

            #     # 바운딩 박스 그리기
            #     result_image2 = draw_bounding_box(image2.copy(), prediction2)
            #     st.image(result_image2, caption="분류 결과 (카메라 2)", use_column_width=True)
                
            #     # 모든 클래스의 확률 표시
            #     st.write("클래스별 확률 (카메라 2):")
            #     for i, prob in enumerate(prediction2):
            #         st.write(f"{labels[i]}: {prob:.2f}")

            # except Exception as e:
            #     st.error(f"예측 중 오류 발생 (카메라 2): {e}")

    
    # CSV 다운로드 버튼
    if 'get_csv' in globals():  # get_csv 함수가 정의되어 있는지 확인
        st.download_button(
            label="CSV 다운로드",
            data=get_csv(),
            file_name="predictions.csv",
            mime="text/csv",
            key="csv_download"
        )


# main() 함수 외부의 코드

if __name__ == "__main__":
    main()

# 데이터베이스 연결 종료
conn.close()

# ------------- 3D Graph 구현 (main 함수의 마지막에 추가) -------------
st.header("3D 그래프")
    
import plotly.graph_objects as go
import numpy as np

# 3D Graph 구현
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
st.plotly_chart(fig)

#------------------- PLC 연결 ----------------------

# PLC 연결 정보
PLC_HOST = "192.168.1.10"  # PLC의 IP 주소
PLC_PORT = 5007  # MELSOFT 통신 포트
PLC_TYPE = "Q"  # PLC 시리즈 (예: Q, L, iQ-R 등)

def connect_plc():
    return Type4E(host=PLC_HOST, port=PLC_PORT, plc_type=PLC_TYPE)

st.title("Mitsubishi PLC 제어")

if st.button("PLC 연결"):
    with connect_plc() as plc:
        st.success("PLC에 연결되었습니다.")
        
        # PLC 상태 읽기
        cpu_model = plc.read_cpu_model()
        st.write(f"CPU 모델: {cpu_model.name}")
        
        # 디바이스 읽기 예시
        d_value = plc.read_dword("D100", 1)[0]
        st.write(f"D100 값: {d_value}")

if st.button("PLC 실행"):
    with connect_plc() as plc:
        plc.remote_run()
        st.success("PLC가 실행 모드로 전환되었습니다.")

if st.button("PLC 정지"):
    with connect_plc() as plc:
        plc.remote_stop()
        st.success("PLC가 정지 모드로 전환되었습니다.")

# 디바이스 쓰기 예시
new_value = st.number_input("D100에 쓸 값", value=0)
if st.button("D100에 값 쓰기"):
    with connect_plc() as plc:
        plc.write_dword("D100", [new_value])
        st.success(f"D100에 {new_value} 값을 썼습니다.")
