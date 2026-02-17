import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
import math
import os
from fpdf import FPDF

# Importación directa de soluciones
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

st.set_page_config(page_title="KININ Motion Tech v40", page_icon="📐", layout="wide")

st.title("📐 KININ Motion Tech - Web Auditor")
st.sidebar.header("Datos de Evaluación")
cliente = st.sidebar.text_input("Empresa", "Cliente Demo")
trabajador = st.sidebar.text_input("Trabajador", "Operario 1")
carga = st.sidebar.number_input("Carga (Kg)", 0.0, 50.0, 0.0)

uploaded_file = st.file_uploader("Subir video para análisis (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    
    # Iniciar motor MediaPipe
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (640, 480))
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)
            
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                
            st_frame.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    st.success("Análisis terminado.")
