import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# Configuración básica
st.set_page_config(page_title="KININ Motion Tech", layout="wide")
st.title("📐 KININ Motion Tech - Auditor Web")

# Carga de video
uploaded_file = st.file_uploader("Sube un video corto (MP4)", type=['mp4', 'mov'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    
    # Motor de IA
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (640, 480))
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(img_rgb)
            
            # Dibujar esqueleto si hay resultados
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            st_frame.image(frame, channels="BGR", use_container_width=True)
            
    cap.release()
    st.success("Análisis finalizado.")
