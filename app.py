import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import time
import pandas as pd
import plotly.express as px
from fpdf import FPDF

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Motion Tech AI - Pro Auditor", layout="wide")

# Estilo personalizado para el Dashboard
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #4b506d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Motion Tech AI | Auditoría Anexo 14")
st.markdown("---")

# --- SIDEBAR: DATOS ANEXO 14 ---
st.sidebar.header("📋 Expediente del Trabajador")
empresa = st.sidebar.text_input("Empresa Principal", "CODELCO")
nombre_trabajador = st.sidebar.text_input("Nombre Trabajador", "Juan Pérez")
rut = st.sidebar.text_input("RUT", "12.345.678-9")
cargo = st.sidebar.text_input("Cargo / Puesto", "Operador de Chancado")
antiguedad = st.sidebar.number_input("Antigüedad (años)", 0, 40, 5)
ciclo_seg = st.sidebar.number_input("Duración Ciclo (seg)", 1, 300, 30)

# --- LÓGICA BIOMECÁNICA ---
def calcular_angulo(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return int(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

# --- INTERFAZ DE TABS ---
tab1, tab2, tab3 = st.tabs(["🎥 Telemetría en Vivo", "📊 Análisis de Riesgo", "📄 Generar EPT"])

with tab1:
    uploaded_file = st.file_uploader("Subir video de evaluación técnica", type=['mp4', 'mov'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        # Almacén de datos para el análisis
        history_hombro = []
        history_tronco = []
        
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils

        with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (854, 480))
                res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    
                    # Ángulos
                    ang_h = calcular_angulo([lm[14].x, lm[14].y], [lm[12].x, lm[12].y], [lm[24].x, lm[24].y])
                    ang_t = 180 - calcular_angulo([lm[12].x, lm[12].y], [lm[24].x, lm[24].y], [lm[26].x, lm[26].y])
                    
                    history_hombro.append(ang_h)
                    history_tronco.append(ang_t)
                    
                    # UI Semáforo
                    color_h = (0, 0, 255) if ang_h > 60 else (0, 255, 0)
                    cv2.putText(frame, f"Hombro: {ang_h} deg", (20, 50), 1, 2, color_h, 3)
                    cv2.putText(frame, f"Tronco: {ang_t} deg", (20, 100), 1, 2, (255, 255, 255), 3)

                st_frame.image(frame, channels="BGR", use_column_width=True)
        
        cap.release()
        st.session_state['h_data'] = history_hombro
        st.session_state['t_data'] = history_tronco
        st.success("Procesamiento completo.")

with tab2:
    if 'h_data' in st.session_state:
        st.subheader("Análisis Estadístico de Posturas")
        h_array = np.array(st.session_state['h_data'])
        
        # Métricas Anexo 14
        critico_h = (h_array > 60).mean() * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Ángulo Máximo Hombro", f"{h_array.max()}°")
        col2.metric("% Tiempo en Riesgo (>60°)", f"{int(critico_h)}%", delta="- Riesgo Alto" if critico_h > 10 else "Riesgo Bajo")
        col3.metric("Frecuencia (Ciclos/min)", f"{int(60/ciclo_seg)}")

        # Gráfico Plotly
        df_plot = pd.DataFrame({"Frame": range(len(h_array)), "Ángulo": h_array})
        fig = px.line(df_plot, x="Frame", y="Ángulo", title="Variación Angular del Hombro durante el Ciclo")
        fig.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Límite Crítico")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    if 'h_data' in st.session_state:
        if st.button("🚀 Generar Informe Técnico de Auditoría"):
            pdf = FPDF()
            pdf.add_page()
            
            # Encabezado Corporativo
            pdf.set_fill_color(30, 33, 48)
            pdf.rect(0, 0, 210, 40, 'F')
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Arial", 'B', 18)
            pdf.cell(0, 15, "INFORME TECNICO BIOMECANICO - ANEXO 14", 0, 1, 'C')
            
            # Datos Generales
            pdf.set_text_color(0, 0, 0)
            pdf.ln(25)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "1. ANTECEDENTES GENERALES", 1, 1, 'L')
            pdf.set_font("Arial", size=10)
            data = [
                ["Empresa:", empresa, "RUT:", rut],
                ["Trabajador:", nombre_trabajador, "Cargo:", cargo],
                ["Antigüedad:", f"{antiguedad} años", "Ciclo:", f"{ciclo_seg} seg"]
            ]
            for row in data:
                pdf.cell(45, 10, row[0], 0)
                pdf.cell(50, 10, row[1], 0)
                pdf.cell(45, 10, row[2], 0)
                pdf.cell(50, 10, row[3], 0, 1)

            # Resultados IA
            pdf.ln(10)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "2. ANALISIS DE RIESGO IA (MOTION TECH)", 1, 1, 'L')
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 10, f"- Flexion Maxima de Hombro: {np.array(st.session_state['h_data']).max()} grados", 0, 1)
            pdf.cell(0, 10, f"- Porcentaje de tiempo sobre el limite (60 deg): {int((np.array(st.session_state['h_data']) > 60).mean()*100)}%", 0, 1)
            pdf.cell(0, 10, f"- Frecuencia de movimientos: {int(60/ciclo_seg)} ciclos por minuto", 0, 1)
            
            # Conclusión Automática
            riesgo = "ALTO" if (np.array(st.session_state['h_data']) > 60).mean() > 0.1 else "BAJO"
            pdf.ln(10)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(200, 0, 0) if riesgo == "ALTO" else pdf.set_text_color(0, 150, 0)
            pdf.cell(0, 10, f"CONCLUSION: NIVEL DE RIESGO ERGONOMICO {riesgo}", 0, 1, 'C')

            output_file = f"EPT_MT_{nombre_trabajador}.pdf"
            pdf.output(output_file)
            with open(output_file, "rb") as f:
                st.download_button("📥 DESCARGAR INFORME OFICIAL", f, file_name=output_file)
