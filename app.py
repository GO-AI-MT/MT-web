import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
import math
import os
from fpdf import FPDF

# --- IMPORTACIÓN ROBUSTA DE MEDIAPIPE ---
from mediapipe.solutions import holistic as mp_holistic
from mediapipe.solutions import drawing_utils as mp_drawing
from mediapipe.solutions import drawing_styles as mp_drawing_styles

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="KININ Motion Tech v40", page_icon="📐", layout="wide")

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007):
        self.x_prev = None; self.dx_prev = 0; self.t_prev = None
        self.min_cutoff = min_cutoff; self.beta = beta
    def filter(self, x, t):
        if self.x_prev is None: self.x_prev = x; self.t_prev = t; return x
        dt = t - self.t_prev; self.t_prev = t
        if dt <= 0: return x
        a_d = (2 * math.pi * dt) / (2 * math.pi * dt + 1)
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = (2 * math.pi * cutoff * dt) / (2 * math.pi * cutoff * dt + 1)
        self.x_prev = a * x + (1 - a) * self.x_prev
        return self.x_prev

# --- INTERFAZ ---
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
    
    # Inicialización de Filtros
    filters = {k: OneEuroFilter() for k in ["h_d", "tronco"]}
    
    # PROCESAMIENTO
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
            
            # Dibujo de malla facial y esqueleto
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                
                # Cálculo de ángulos básicos
                lm = results.pose_landmarks.landmark
                def get_a(a,b,c):
                    v1 = np.array([lm[a].x - lm[b].x, lm[a].y - lm[b].y])
                    v2 = np.array([lm[c].x - lm[b].x, lm[c].y - lm[b].y])
                    return int(np.degrees(np.arccos(np.clip(np.dot(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)), -1, 1))))
                
                h_d = get_a(14,12,24)
                tr = get_a(11,23,25)
                
                cv2.putText(frame, f"Hombro: {h_d}deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Tronco: {tr}deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            st_frame.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    st.success("Análisis terminado.")
