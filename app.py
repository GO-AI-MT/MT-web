import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import time
from fpdf import FPDF

# --- CONFIGURACIÓN E INTERFAZ ---
st.set_page_config(page_title="KININ Motion Tech Pro", layout="wide")
st.title("📐 KININ Motion Tech - Análisis Profesional")

# Importación de soluciones
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Sidebar para datos del reporte
st.sidebar.header("📝 Datos del Informe")
empresa = st.sidebar.text_input("Empresa", "KININ")
trabajador = st.sidebar.text_input("Nombre del Trabajador", "Operario")
carga_kg = st.sidebar.number_input("Carga Manipulada (Kg)", 0.0, 50.0, 0.0)

# Función para calcular ángulos
def calcular_angulo(p1, p2, p3):
    """Calcula el ángulo entre tres puntos (x, y)."""
    a = np.array(p1) # Punto extremo 1
    b = np.array(p2) # Vértice
    c = np.array(p3) # Punto extremo 2
    
    # Vectores
    ba = a - b
    bc = c - b
    
    # Fórmula: cos(theta) = (ba . bc) / (|ba| * |bc|)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return int(np.degrees(angle))

# --- PROCESAMIENTO DE VIDEO ---
uploaded_file = st.file_uploader("Sube el video de la evaluación (MP4/MOV)", type=['mp4', 'mov'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    
    # Variables de seguimiento
    max_hombro_d = 0
    max_tronco = 0
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (640, 480))
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(img_rgb)
            
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                # Dibujar esqueleto
                mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                
                # Obtener coordenadas (x, y)
                hombro_d = [lm[12].x, lm[12].y]
                codo_d = [lm[14].x, lm[14].y]
                cadera_d = [lm[24].x, lm[24].y]
                rodilla_d = [lm[26].x, lm[26].y]
                
                # Cálculos
                # 1. Flexión de Hombro (Ángulo entre Codo-Hombro-Cadera)
                ang_hombro = calcular_angulo(codo_d, hombro_d, cadera_d)
                # 2. Flexión de Tronco (Ángulo entre Hombro-Cadera-Rodilla)
                ang_tronco = 180 - calcular_angulo(hombro_d, cadera_d, rodilla_d)
                
                # Actualizar máximos
                max_hombro_d = max(max_hombro_d, ang_hombro)
                max_tronco = max(max_tronco, ang_tronco)
                
                # UI en el video
                cv2.rectangle(frame, (0, 0), (220, 80), (0, 0, 0), -1)
                cv2.putText(frame, f"Hombro: {ang_hombro} deg", (10, 30), 1, 1.2, (0, 255, 0), 2)
                cv2.putText(frame, f"Tronco: {ang_tronco} deg", (10, 65), 1, 1.2, (0, 255, 0), 2)

            st_frame.image(frame, channels="BGR", use_column_width=True)
            
    cap.release()
    st.success("✅ Análisis finalizado.")
    
    # --- SECCIÓN DE RESULTADOS Y PDF ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.metric("Máx Flexión Hombro", f"{max_hombro_d}°")
    col2.metric("Máx Flexión Tronco", f"{max_tronco}°")
    
    # Generar PDF
    if st.button("📄 Generar Informe PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "INFORME TECNICO DE EVALUACION BIOMECANICA", 0, 1, 'C')
        pdf.ln(10)
        
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Empresa: {empresa}", 0, 1)
        pdf.cell(0, 10, f"Trabajador: {trabajador}", 0, 1)
        pdf.cell(0, 10, f"Carga: {carga_kg} Kg", 0, 1)
        pdf.cell(0, 10, f"Fecha: {time.strftime('%d/%m/%Y')}", 0, 1)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "RESULTADOS MAXIMOS DETECTADOS:", 0, 1)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"- Angulo Max Hombro: {max_hombro_d} grados", 0, 1)
        pdf.cell(0, 10, f"- Angulo Max Tronco: {max_tronco} grados", 0, 1)
        
        pdf_output = f"Informe_{trabajador}.pdf"
        pdf.output(pdf_output)
        
        with open(pdf_output, "rb") as f:
            st.download_button("⬇️ Descargar Informe", f, file_name=pdf_output)
        os.remove(pdf_output)

    os.remove(tfile.name)
