import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Streamlit 앱 설정
st.set_page_config(
    page_title='취업 성공 예측 시스템',
    page_icon=':briefcase:',
)

# -------------------------------------------------------------------------
# 모델 로드 함수 정의

@st.cache_resource
def load_model(uploaded_file):
    """Joblib 파일에서 모델을 로드"""
    try:
        model = joblib.load(uploaded_file)
        return model
    except Exception as e:
        st.error(f"모델 로드 중 오류가 발생했습니다: {e}")
        return None

# -------------------------------------------------------------------------
# 앱 구현

st.title(':briefcase: 취업 성공 예측 시스템')
st.write("""
이 시스템은 사용자가 입력한 데이터를 기반으로 취업 성공 가능성을 예측합니다.
업로드된 모델 파일을 사용하여 예측을 수행합니다.
""")

# 모델 파일 업로드
uploaded_model = st.file_uploader("예측 모델 (.joblib 파일)을 업로드하세요", type=['joblib'])

if uploaded_model:
    # 모델 로드
    model = load_model(uploaded_model)

    if model is not None:
        st.success("모델이 성공적으로 로드되었습니다!")

        # 사용자 입력 섹션
        st.header("입력 데이터")
        st.write("아래 입력 필드를 채워 예측을 진행하세요.")

        # 사용자 입력 값
        age = st.number_input("나이", min_value=18, max_value=65, step=1, format="%d")
        education_level = st.selectbox("학력 수준", ["고졸", "대졸", "석사", "박사"])
        work_experience = st.number_input("경력 (년)", min_value=0, max_value=40, step=1, format="%d")
        skills_score = st.slider("기술 점수", min_value=0, max_value=100, value=50)

        # 학력 수준 변환
        education_map = {"고졸": 1, "대졸": 2, "석사": 3, "박사": 4}
        education_encoded = education_map.get(education_level, 1)

        # 입력 데이터를 배열로 변환
        input_data = np.array([[age, education_encoded, work_experience, skills_score]])

        # 예측 수행
        if st.button("예측"):
            try:
                prediction = model.predict(input_data)
                success_probability = model.predict_proba(input_data)[0][1] * 100

                st.subheader("예측 결과")
                st.write(f"취업 성공 가능성: **{success_probability:.2f}%**")
                st.write("예측 값:", "취업 성공" if prediction[0] == 1 else "취업 실패")

            except Exception as e:
                st.error(f"예측 중 오류가 발생했습니다: {e}")
else:
    st.info("모델 파일을 업로드해주세요.")
