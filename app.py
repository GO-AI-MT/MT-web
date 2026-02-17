import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
import math
import os
from fpdf import FPDF
from datetime import datetime

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="KININ Motion Tech", page_icon="📐", layout="wide")

# --- MOTORES MATEMÁTICOS SIMPLIFICADOS PARA WEB ---
class REBA_Engine:
    @staticmethod
    def calcular(vals, carga):
        # Lógica REBA Web
        score = 1
        if vals['tronco'] > 20: score += 2
        if vals['cuello'] > 20: score += 1
        if vals['h_d'] > 45 or vals['h_i'] > 45: score += 1
        if carga > 5: score += 1
        
        riesgo = "BAJO"
        if score >= 8: riesgo = "ALTO"
        elif score >= 4: riesgo = "MEDIO"
        return score, riesgo

def crear_pdf_simple(tr, cl, carga, riesgo, stats):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "INFORME KININ MOTION TECH (WEB)", 0, 1, 'C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Trabajador: {tr}", 0, 1)
    pdf.cell(0, 10, f"Empresa: {cl}", 0, 1)
    pdf.cell(0, 10, f"Carga: {carga} kg", 0, 1)
    pdf.ln(5)
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(0, 10, f"RESULTADO GLOBAL: RIESGO {riesgo}", 1, 1, 'C', True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Estadisticas:\n- Flexion Hombro D: {stats['h_d']} deg\n- Flexion Tronco: {stats['tronco']} deg")
    
    # Guardar en temporal para descarga web
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_file.name)
    return tmp_file.name

# --- INTERFAZ GRÁFICA WEB ---
st.title("📐 KININ MOTION TECH - WEB AUDITOR")
st.markdown("---")

# Barra Lateral de Datos
with st.sidebar:
    st.header("1. Datos del Puesto")
    cliente = st.text_input("Empresa", "Cliente Demo")
    trabajador = st.text_input("Trabajador", "Operario 1")
    modo = st.selectbox("Modo", ["OPERATIVO (Pie)", "ADMINISTRATIVO (Silla)"])
    
    st.header("2. Carga Física")
    carga = st.number_input("Peso Carga (Kg)", 0.0, 50.0, 0.0)
    borg = st.slider("Escala Borg (0-10)", 0, 10, 2)

# Área de Carga de Video
uploaded_file = st.file_uploader("📂 Arrastra tu video aquí (MP4/AVI)", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # Guardar video temporalmente
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("Procesando video con IA...")
        video_placeholder = st.empty()
        
    with col2:
        st.write("### 📊 Telemetría en Vivo")
        metric_h = st.empty()
        metric_t = st.empty()
        metric_r = st.empty()

    # Iniciar Motores
    cap = cv2.VideoCapture(tfile.name)
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    max_h_d = 0
    max_tronco = 0
    riesgo_final = "BAJO"

    with mp_holistic.Holistic(min_detection_confidence=0.5, model_complexity=1) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Procesar Frame
            h, w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(img_rgb)
            
            # Dibujar Esqueleto
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                
                # Calcular Ángulos Básicos
                lm = res.pose_landmarks.landmark
                def get_ang(a,b,c):
                    return int(abs(math.degrees(math.atan2(lm[c].y-lm[b].y, lm[c].x-lm[b].x) - math.atan2(lm[a].y-lm[b].y, lm[a].x-lm[b].x))))
                
                ang_h_d = get_ang(14,12,24)
                ang_tronco = get_ang(11,23,25)
                
                # Actualizar Maximos
                if ang_h_d > max_h_d: max_h_d = ang_h_d
                if ang_tronco > max_tronco: max_tronco = ang_tronco
                
                # Actualizar Métricas
                metric_h.metric("Hombro Der", f"{ang_h_d}°")
                metric_t.metric("Tronco", f"{ang_tronco}°")
                
                # REBA Live
                score, riesgo = REBA_Engine.calcular({'h_d':ang_h_d, 'h_i':0, 'tronco':ang_tronco, 'cuello':0}, carga)
                riesgo_final = riesgo
                
                color_r = (0,255,0)
                if riesgo == "ALTO": color_r = (0,0,255)
                elif riesgo == "MEDIO": color_r = (0,165,255)
                
                metric_r.markdown(f"**Riesgo Actual:** :{('red' if riesgo=='ALTO' else 'orange' if riesgo=='MEDIO' else 'green')}[{riesgo}]")
                
                # Overlay en Video
                cv2.putText(frame, f"REBA: {score} ({riesgo})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_r, 2)

            # Mostrar en Web
            video_placeholder.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    st.success("✅ Análisis Completado")
    
    # Generar Reporte PDF
    if st.button("📄 Descargar Informe Pericial (PDF)"):
        pdf_path = crear_pdf_simple(trabajador, cliente, carga, riesgo_final, {'h_d': max_h_d, 'tronco': max_tronco})
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Guardar PDF", f, file_name=f"Informe_Kinin_{trabajador}.pdf")

else:
    st.info("Esperando video... Sube uno en el panel de arriba.")