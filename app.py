import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

# Configuración básica de la página
st.set_page_config(page_title="KININ Motion Tech", layout="wide")
st.title("📐 KININ Motion Tech - Auditor Web")

# Importación de soluciones de MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Barra lateral para datos
st.sidebar.header("Configuración")
carga = st.sidebar.number_input("Carga (Kg)", 0.0, 50.0, 0.0)

# Carga de video
uploaded_file = st.file_uploader("Sube un video corto (MP4/MOV)", type=['mp4', 'mov'])

if uploaded_file:
    # Guardar video temporal
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    
    # Motor de IA (Holistic)
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Redimensionar para velocidad en la web
            frame = cv2.resize(frame, (640, 480))
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(img_rgb)
            
            # Dibujar esqueleto (Pose)
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    res.pose_landmarks, 
                    mp_holistic.POSE_CONNECTIONS
                )
            
            # MOSTRAR EN PANTALLA
            # Cambiamos 'use_container_width' por 'use_column_width' para v1.31.0
            st_frame.image(frame, channels="BGR", use_column_width=True)
            
    cap.release()
    st.success("✅ Análisis finalizado con éxito.")
    os.remove(tfile.name)
else:
    st.info("Esperando video para procesar...")
