import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
import time
from PIL import ImageFont, ImageDraw, Image

# Mediapipe 초기화
mp_holistic = mp.solutions.holistic

# 단어 목록
actions = np.array(['None', '계산', '고맙다', '괜찮다', '기다리다', '나', '네', '다음',
                    '달다', '더', '도착', '돈', '또', '맵다', '먼저', '무엇', '물', '물음',
                    '부탁', '사람', '수저', '시간', '아니요', '어디', '얼마', '예약', '오다',
                    '우리', '음식', '이거', '인기', '있다', '자리', '접시', '제일', '조금',
                    '주문', '주세요', '짜다', '책', '추천', '화장실', '확인'])

# 모델 구조 정의
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# 학습된 가중치 불러오기
model.load_weights('./model.h5')


# 결과 저장용
sequence = []
threshold = 0.75

# 키포인트 추출 함수
def extract_keypoints(results):
    lh = np.array([[res.x*3, res.y*3, res.z*3] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x*3, res.y*3, res.z*3] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# 웹캠 연결
cap = cv2.VideoCapture(0)

font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"  # macOS 기준, 필요시 수정

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mediapipe 추론
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 키포인트 추출 및 시퀀스 추가
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        if len(sequence) > 30:
            sequence.pop(0)

        # 30프레임마다 예측
        if len(sequence) == 30:
            input_seq = np.expand_dims(sequence, axis=0)
            res = model.predict(input_seq)[0]

            if res[np.argmax(res)] > threshold:
                predicted_word = actions[np.argmax(res)]
                if predicted_word != 'None':
                    print("예측:", predicted_word)
                else:
                    predicted_word = ""
            else:
                predicted_word = ""

        # 시각화
        cv2.rectangle(image, (0, 0), (640, 60), (245, 117, 16), -1)
        display_text = predicted_word if 'predicted_word' in locals() else ""

        # PIL 이미지로 변환
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.truetype(font_path, 32)
        draw.text((10, 15), f"단어: {display_text}", font=font, fill=(255, 255, 255))
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


        # 출력
        cv2.imshow('Webcam Inference', image)

        # 종료 조건
        key = cv2.waitKey(10)
        if key == 27 or key == ord('q'):  # ESC 또는 'q' 누르면 종료
            break

# 종료
cap.release()
cv2.destroyAllWindows()