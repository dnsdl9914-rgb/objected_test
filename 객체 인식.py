import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import time
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# 페이지 설정
st.set_page_config(
    page_title="YOLO 객체 인식",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS 스타일
st.markdown("""
<style>
    /* 전체 배경 그라디언트 */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* 사이드바 스타일 */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-right: 2px solid rgba(255, 255, 255, 0.3);
        box-shadow: 2px 0 20px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown h1 {
        color: #667eea;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* 메인 컨텐츠 영역 */
    .main .block-container {
        padding: 2rem 3rem;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        margin: 2rem auto;
        backdrop-filter: blur(10px);
    }
    
    /* 제목 스타일 */
    h1 {
        color: #667eea !important;
        font-weight: 800 !important;
        text-align: center !important;
        font-size: 3rem !important;
        margin-bottom: 1rem !important;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.1);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* 서브 헤더 스타일 */
    h2, h3 {
        color: #764ba2 !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Primary 버튼 */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 5px 15px rgba(245, 87, 108, 0.4);
    }
    
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.6);
    }
    
    /* 슬라이더 스타일 */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 메트릭 카드 스타일 */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #667eea !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 600 !important;
        color: #764ba2 !important;
    }
    
    /* 정보 박스 스타일 */
    .stAlert {
        border-radius: 15px;
        border: none;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Success 박스 */
    .stSuccess {
        background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
        color: white;
    }
    
    /* Warning 박스 */
    .stWarning {
        background: linear-gradient(135deg, #FFD26F 0%, #FFA500 100%);
        color: white;
    }
    
    /* Info 박스 */
    .stInfo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 카메라 입력 영역 */
    [data-testid="stCameraInput"] {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        border: 3px solid rgba(102, 126, 234, 0.3);
    }
    
    /* 이미지 스타일 */
    img {
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* 구분선 스타일 */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, #667eea, transparent);
    }
    
    /* Select box 스타일 */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    
    /* 애니메이션 효과 */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .element-container {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* 스크롤바 스타일 */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2 0%, #667eea 100%);
    }
    
    /* 카드 스타일 효과 */
    .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'model' not in st.session_state:
    st.session_state.model = None
if 'is_counting' not in st.session_state:
    st.session_state.is_counting = False
if 'countdown_start' not in st.session_state:
    st.session_state.countdown_start = None
if 'selected_person' not in st.session_state:
    st.session_state.selected_person = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'detection_info' not in st.session_state:
    st.session_state.detection_info = None
if 'camera_key' not in st.session_state:
    st.session_state.camera_key = 0
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'captured_result' not in st.session_state:
    st.session_state.captured_result = None

# 사이드바 설정
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='font-size: 2rem; color: #667eea;'>⚙️ 설정</h1>
            <p style='color: #764ba2; font-size: 0.9rem;'>모델과 파라미터를 조정하세요</p>
        </div>
    """, unsafe_allow_html=True)
    
    model_name = st.selectbox(
        "모델 선택",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        index=0
    )
    
    confidence = st.slider(
        "탐지 임계값 (Confidence)",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.1
    )
    
    countdown_time = st.slider(
        "카운트다운 시간 (초)",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )
    
    img_size = st.selectbox(
        "이미지 크기",
        [320, 640, 1280],
        index=1
    )
    
    # 모델 로드
    if st.button("모델 로드", type="primary"):
        with st.spinner("모델을 로드하는 중..."):
            try:
                # 현재 스크립트 파일의 디렉토리 경로 찾기
                script_dir = Path(__file__).parent
                model_path = script_dir / model_name
                
                # 로컬 파일이 있으면 로컬 경로 사용, 없으면 원래 이름 사용 (자동 다운로드)
                if model_path.exists():
                    st.session_state.model = YOLO(str(model_path))
                    st.success(f"✅ {model_name} 모델이 로드되었습니다! (로컬 파일 사용)")
                else:
                    st.session_state.model = YOLO(model_name)
                    st.success(f"✅ {model_name} 모델이 로드되었습니다!")
            except Exception as e:
                st.error(f"모델 로드 실패: {e}")

# 메인 영역
st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='font-size: 3.5rem; margin-bottom: 0.5rem;'>
            🎯 YOLO 실시간 객체 인식
        </h1>
        <p style='font-size: 1.2rem; color: #764ba2; font-weight: 500;'>
            AI로 순간을 포착하고 분석하세요 ✨
        </p>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

# 모델이 로드되지 않았으면 경고
if st.session_state.model is None:
    st.warning("⚠️ 사이드바에서 모델을 먼저 로드해주세요.")
    st.stop()

# 카메라 입력
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 15px; margin-bottom: 1rem; text-align: center;'>
            <h2 style='color: white; margin: 0; font-size: 1.8rem;'>📹 카메라</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # 선택이 완료되어 결과를 표시하는 경우
    if st.session_state.selected_person is not None and st.session_state.captured_image is not None:
        # 촬영된 이미지에 선택된 사람 표시
        annotated_img = st.session_state.captured_image.copy()
        
        x1, y1, x2, y2 = st.session_state.selected_person['bbox']
        # 빨간색 두꺼운 박스 그리기
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 5)
        # "축하합니다." 텍스트 표시 (한글 지원을 위해 PIL 사용)
        text = "축하합니다."
        # numpy 배열을 PIL Image로 변환
        pil_img = Image.fromarray(annotated_img)
        draw = ImageDraw.Draw(pil_img)
        
        # 폰트 설정 (시스템 기본 폰트 사용, 없으면 기본 폰트)
        try:
            # Windows의 경우
            font = ImageFont.truetype("malgun.ttf", 40)
        except:
            try:
                # macOS의 경우
                font = ImageFont.truetype("/System/Library/Fonts/AppleGothic.ttf", 40)
            except:
                try:
                    # 다른 경로 시도
                    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/AppleGothic.ttf", 40)
                except:
                    # 기본 폰트 사용
                    font = ImageFont.load_default()
        
        # 텍스트 크기 계산
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 텍스트 배경 그리기 (빨간색 네모칸)
        padding = 10
        draw.rectangle([x1, y1 - text_height - padding * 2, x1 + text_width + padding * 2, y1], 
                      fill=(255, 0, 0), outline=None)
        
        # 텍스트 그리기
        draw.text((x1 + padding, y1 - text_height - padding), text, 
                 fill=(255, 255, 255), font=font)
        
        # PIL Image를 numpy 배열로 다시 변환
        annotated_img = np.array(pil_img)
        
        st.image(annotated_img, use_container_width=True, channels="RGB")
        st.success("🎉 선택 완료! 리셋 버튼을 눌러 다시 시작하세요.")
    
    # 카운트다운 중일 때 (사진은 촬영되었지만 아직 선택 전)
    elif st.session_state.is_counting and st.session_state.captured_image is not None:
        current_time = time.time()
        elapsed = current_time - st.session_state.countdown_start
        remaining = countdown_time - elapsed
        
        if remaining > 0:
            # 카운트다운 표시
            annotated_img = st.session_state.captured_image.copy()
            countdown_text = f"Selection in: {remaining:.1f}s"
            cv2.putText(annotated_img, countdown_text, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            st.image(annotated_img, use_container_width=True, channels="RGB")
            time.sleep(0.1)
            st.rerun()
        else:
            # 시간이 지나면 사람 선택
            st.session_state.is_counting = False
            
            # person 클래스만 필터링 (클래스 ID: 0)
            persons = []
            if st.session_state.captured_result.boxes is not None:
                for box in st.session_state.captured_result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0:  # person 클래스
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        persons.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': conf
                        })
            
            # 사람이 있으면 랜덤으로 한 명 선택
            if persons:
                st.session_state.selected_person = random.choice(persons)
            else:
                st.session_state.selected_person = None
                st.warning("⚠️ 인식된 사람이 없습니다.")
            
            st.rerun()
    
    # 카메라 입력 (초기 상태)
    else:
        camera_input = st.camera_input("'Take Photo' 버튼을 눌러 시작하세요!", key=f"camera_{st.session_state.camera_key}")
        
        if camera_input is not None:
            st.session_state.camera_active = True
            # 이미지를 numpy 배열로 변환
            bytes_data = camera_input.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            
            # YOLO 추론
            results = st.session_state.model.predict(
                source=cv2_img,
                conf=confidence,
                imgsz=img_size,
                verbose=False
            )
            
            result = results[0]
            annotated_img = result.plot()
            
            # 촬영된 이미지와 결과 저장
            st.session_state.captured_image = annotated_img
            st.session_state.captured_result = result
            
            # 탐지 정보 저장
            class_counts = {}
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            st.session_state.detection_info = class_counts
            
            # 사진 촬영과 동시에 카운트다운 시작
            st.session_state.is_counting = True
            st.session_state.countdown_start = time.time()
            
            st.rerun()
        else:
            st.session_state.camera_active = False
            st.info("📷 'Take Photo' 버튼을 눌러 시작하세요!")

with col2:
    st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1rem; border-radius: 15px; margin-bottom: 1rem; text-align: center;'>
            <h2 style='color: white; margin: 0; font-size: 1.8rem;'>🎮 컨트롤</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # 리셋 버튼
    if st.button("🔄 리셋", type="primary", use_container_width=True):
        st.session_state.is_counting = False
        st.session_state.selected_person = None
        st.session_state.countdown_start = None
        st.session_state.camera_active = False
        st.session_state.detection_info = None
        st.session_state.captured_image = None
        st.session_state.captured_result = None
        # 카메라 입력을 초기화하기 위해 키 변경
        st.session_state.camera_key += 1
        st.info("🔄 상태가 초기화되었습니다. 다시 촬영해주세요.")
        st.rerun()
    
    st.markdown("---")
    
    # 상태 표시
    st.markdown("""
        <div style='background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%); 
                    padding: 0.8rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;'>
            <h3 style='color: white; margin: 0; font-size: 1.4rem;'>📊 상태</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # 진행 상태
    if st.session_state.selected_person is not None:
        st.success("✅ 선택 완료!")
    elif st.session_state.is_counting:
        st.info("⏱️ 카운트다운 진행 중...")
    elif st.session_state.captured_image is not None:
        st.info("⏳ 처리 중...")
    else:
        st.warning("📸 사진을 촬영해주세요")
    
    # 카운트다운 상태
    if st.session_state.is_counting:
        elapsed = time.time() - st.session_state.countdown_start
        remaining = max(0, countdown_time - elapsed)
        st.metric("남은 시간", f"{remaining:.1f}초")
        st.progress(1 - (remaining / countdown_time))
    else:
        st.metric("남은 시간", "-")
    
    # 선택 결과
    if st.session_state.selected_person is not None:
        st.metric("신뢰도", f"{st.session_state.selected_person['confidence']:.2%}")
    
    st.markdown("---")
    
    # 탐지 정보
    if st.session_state.detection_info is not None:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                        padding: 0.8rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;'>
                <h3 style='color: #764ba2; margin: 0; font-size: 1.4rem;'>📈 탐지 정보</h3>
            </div>
        """, unsafe_allow_html=True)
        if len(st.session_state.detection_info) > 0:
            for cls_name, count in st.session_state.detection_info.items():
                st.metric(cls_name, count)
            
            # 사람 수
            person_count = st.session_state.detection_info.get('person', 0)
            if person_count > 0:
                st.success(f"👤 {person_count}명의 사람이 감지되었습니다.")
        else:
            st.info("감지된 객체가 없습니다.")
    else:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                        padding: 0.8rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;'>
                <h3 style='color: #764ba2; margin: 0; font-size: 1.4rem;'>📈 탐지 정보</h3>
            </div>
        """, unsafe_allow_html=True)
        st.info("카메라를 촬영하면 탐지 정보가 표시됩니다.")

# 하단 정보
st.markdown("---")
st.markdown("""
<div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
            padding: 2rem; border-radius: 20px; margin-top: 2rem; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);'>
    <h3 style='color: #667eea; text-align: center; margin-bottom: 1.5rem; font-size: 2rem;'>
        🎯 사용 방법
    </h3>
    <div style='background: white; padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);'>
        <h4 style='color: #667eea; margin-bottom: 0.5rem;'>1️⃣ 모델 로드</h4>
        <p style='color: #555; margin-left: 1.5rem;'>사이드바에서 모델을 선택하고 <strong>"모델 로드"</strong> 버튼을 클릭하세요.</p>
    </div>
    <div style='background: white; padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);'>
        <h4 style='color: #764ba2; margin-bottom: 0.5rem;'>2️⃣ 사진 촬영</h4>
        <p style='color: #555; margin-left: 1.5rem;'>카메라 입력란에서 <strong>"Take Photo"</strong> 버튼을 누르세요.</p>
        <p style='color: #888; margin-left: 1.5rem; font-size: 0.9rem;'>→ 사진 촬영과 동시에 자동으로 카운트다운이 시작됩니다!</p>
    </div>
    <div style='background: white; padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);'>
        <h4 style='color: #667eea; margin-bottom: 0.5rem;'>3️⃣ 자동 선택</h4>
        <p style='color: #555; margin-left: 1.5rem;'>카운트다운이 끝나면 인식된 사람 중 랜덤으로 1명이 선택됩니다.</p>
    </div>
    <div style='background: white; padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);'>
        <h4 style='color: #764ba2; margin-bottom: 0.5rem;'>4️⃣ 결과 확인</h4>
        <p style='color: #555; margin-left: 1.5rem;'>선택된 사람은 빨간 박스와 <strong>"축하합니다."</strong> 메시지로 표시됩니다.</p>
    </div>
    <div style='background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);'>
        <h4 style='color: #667eea; margin-bottom: 0.5rem;'>5️⃣ 리셋</h4>
        <p style='color: #555; margin-left: 1.5rem;'><strong>"리셋"</strong> 버튼을 누르면 처음부터 다시 시작할 수 있습니다.</p>
    </div>
    <div style='background: linear-gradient(135deg, #FFD26F 0%, #FFA500 30%); 
                padding: 1rem; border-radius: 15px; margin-top: 1.5rem; text-align: center;'>
        <p style='color: white; margin: 0; font-weight: 600; font-size: 1.1rem;'>
            💡 <strong>팁:</strong> 사이드바에서 카운트다운 시간을 조절할 수 있습니다 (1~10초)
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

