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
cargo = st.sidebar.text_input("Cargo / Puesto", "Operador de CAEX").strip() # .strip() para quitar espacios
antiguedad = st.sidebar.number_input("Antigüedad (años)", 0, 40, 5)
ciclo_seg = st.sidebar.number_input("Duración Ciclo (seg)", 1, 300, 30)

# --- LÓGICA BIOMECÁNICA ---
def calcular_angulo(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return int(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

# --- MOTOR DE RECOMENDACIONES INTELIGENTES ---
def generar_recomendaciones(cargo_input, riesgo_h, riesgo_t):
    recomendaciones = []
    cargo_lower = cargo_input.lower()
    
    # Categorización simple del cargo
    categoria = "general"
    if any(word in cargo_lower for word in ["secretaria", "administrativo", "oficina", "analista"]):
        categoria = "oficina"
    elif any(word in cargo_lower for word in ["operador", "conductor", "chofer", "maquinaria", "caex"]):
        categoria = "maquinaria"
    elif any(word in cargo_lower for word in ["mecánico", "eléctrico", "mantenedor", "obra", "construcción"]):
        categoria = "mantenimiento"

    # Recomendaciones por segmento y categoría
    # --- HOMBRO ---
    if riesgo_h == "ALTO":
        if categoria == "oficina":
            recomendaciones.append("HOMBRO (Alto Riesgo): Revisar altura de silla y monitor. El teclado y mouse deben estar al mismo nivel para evitar elevación mantenida. Implementar pausas con estiramientos de cuello y hombros.")
        elif categoria == "maquinaria":
            recomendaciones.append("HOMBRO (Alto Riesgo): Evaluar la posición de los controles y palancas en la cabina. Deben estar al alcance sin necesidad de elevar o abducir los brazos frecuentemente. Revisar el ajuste del asiento.")
        elif categoria == "mantenimiento":
            recomendaciones.append("HOMBRO (Alto Riesgo): Evitar trabajos con brazos sobre el nivel de los hombros por tiempos prolongados. Utilizar plataformas o herramientas de extensión. Fomentar la alternancia de brazos.")
        else: # General
            recomendaciones.append("HOMBRO (Alto Riesgo): Rediseñar la tarea para evitar alcances frecuentes o sostenidos por encima de la cabeza. Acercar las herramientas y materiales al cuerpo.")
    elif riesgo_h == "MEDIO":
         recomendaciones.append("HOMBRO (Riesgo Medio): Monitorear la postura. Fomentar micro-pausas para relajar la musculatura del hombro. Evaluar si es posible rotar tareas para reducir la exposición.")

    # --- TRONCO ---
    if riesgo_t == "ALTO":
        if categoria == "oficina":
             recomendaciones.append("TRONCO (Alto Riesgo): Verificar el soporte lumbar de la silla. Evitar giros del tronco al alcanzar objetos; usar una silla giratoria. La pantalla debe estar frente al usuario para evitar torsiones de cuello y tronco.")
        elif categoria == "maquinaria":
             recomendaciones.append("TRONCO (Alto Riesgo): Crítico revisar el sistema de suspensión y soporte lumbar del asiento de la cabina para reducir la vibración y mejorar la postura. Asegurar que los espejos y visores no fuercen torsiones.")
        elif categoria == "mantenimiento":
             recomendaciones.append("TRONCO (Alto Riesgo): Evitar posturas de flexión extrema o torsión al manipular cargas o usar herramientas. Capacitar en técnicas de levantamiento seguro. Usar ayudas mecánicas para cargas pesadas.")
        else: # General
             recomendaciones.append("TRONCO (Alto Riesgo): Implementar ayudas mecánicas para el manejo de cargas. Modificar la altura de las superficies de trabajo para evitar flexiones pronunciadas. Evitar la combinación de flexión y torsión.")
    elif riesgo_t == "MEDIO":
        recomendaciones.append("TRONCO (Riesgo Medio): Fomentar la higiene postural. Realizar pausas activas enfocadas en la movilidad y fortalecimiento de la zona lumbar. Revisar la organización del puesto para minimizar alcances lejanos.")

    if not recomendaciones:
        recomendaciones.append("Mantener las buenas prácticas actuales. Reforzar la capacitación en autocuidado y ergonomía preventiva.")
        
    return recomendaciones

# --- INTERFAZ DE TABS ---
tab1, tab2, tab3 = st.tabs(["🎥 Telemetría en Vivo", "📊 Análisis de Riesgo", "📄 Generar EPT"])

with tab1:
    uploaded_file = st.file_uploader("Subir video de evaluación técnica", type=['mp4', 'mov'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        # --- DETECCIÓN DE ORIENTACIÓN ---
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        is_vertical = height > width
        
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
                
                # Redimensionar manteniendo proporción
                if is_vertical:
                    new_height = 640
                    new_width = int(width * (new_height / height))
                    frame = cv2.resize(frame, (new_width, new_height))
                else:
                    new_width = 640
                    new_height = int(height * (new_width / width))
                    frame = cv2.resize(frame, (new_width, new_height))
                    
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
                    # Ajuste posición texto según orientación
                    text_x = 20 if not is_vertical else 10
                    font_scale = 1 if not is_vertical else 0.7
                    thickness = 2
                    
                    cv2.putText(frame, f"Hombro: {ang_h} deg", (text_x, 50), 1, font_scale, color_h, thickness)
                    cv2.putText(frame, f"Tronco: {ang_t} deg", (text_x, 100), 1, font_scale, (255, 255, 255), thickness)

                # Mostrar video con la orientación correcta
                if is_vertical:
                     st_frame.image(frame, channels="BGR", use_column_width=False, width=new_width)
                else:
                     st_frame.image(frame, channels="BGR", use_column_width=True)
        
        cap.release()
        st.session_state['h_data'] = history_hombro
        st.session_state['t_data'] = history_tronco
        st.success("Procesamiento completo.")

with tab2:
    if 'h_data' in st.session_state:
        st.subheader("Análisis Estadístico de Posturas")
        h_array = np.array(st.session_state['h_data'])
        t_array = np.array(st.session_state['t_data'])
        
        # Métricas Anexo 14
        critico_h = (h_array > 60).mean() * 100
        critico_t = (t_array > 60).mean() * 100 # Umbral ejemplo tronco
        
        # Determinar niveles de riesgo globales para recomendaciones
        riesgo_h_global = "ALTO" if critico_h > 10 else ("MEDIO" if critico_h > 5 else "BAJO")
        riesgo_t_global = "ALTO" if critico_t > 10 else ("MEDIO" if critico_t > 5 else "BAJO")
        st.session_state['riesgo_h'] = riesgo_h_global
        st.session_state['riesgo_t'] = riesgo_t_global

        col1, col2, col3 = st.columns(3)
        col1.metric("Ángulo Máx. Hombro", f"{h_array.max()}°")
        col2.metric("% Tiempo Hombro >60°", f"{int(critico_h)}%", delta="- Riesgo Alto" if riesgo_h_global == "ALTO" else "Riesgo Controlado")
        col3.metric("Frecuencia (Ciclos/min)", f"{int(60/ciclo_seg)}")
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Ángulo Máx. Tronco", f"{t_array.max()}°")
        col5.metric("% Tiempo Tronco >60°", f"{int(critico_t)}%", delta="- Riesgo Alto" if riesgo_t_global == "ALTO" else "Riesgo Controlado")
        col6.write("") # Espacio

        # Gráfico Plotly
        df_plot = pd.DataFrame({"Frame": range(len(h_array)), "Hombro": h_array, "Tronco": t_array})
        fig = px.line(df_plot, x="Frame", y=["Hombro", "Tronco"], title="Variación Angular durante el Ciclo")
        fig.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Límite Crítico")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    if 'h_data' in st.session_state and 'riesgo_h' in st.session_state:
        if st.button("🚀 Generar Informe Técnico de Auditoría"):
            # Generar recomendaciones
            recoms = generar_recomendaciones(cargo, st.session_state['riesgo_h'], st.session_state['riesgo_t'])
            
            pdf = FPDF()
            pdf.add_page()
            
            # Encabezado Corporativo
            pdf.set_fill_color(30, 33, 48)
            pdf.rect(0, 0, 210, 40, 'F')
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Arial", 'B', 18)
            pdf.cell(0, 15, "INFORME TECNICO BIOMECANICO - ANEXO 14", 0, 1, 'C')
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, "Estudio de Puesto de Trabajo (EPT) con Inteligencia Artificial", 0, 1, 'C')
            
            # Datos Generales
            pdf.set_text_color(0, 0, 0)
            pdf.ln(20)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_fill_color(200, 220, 255)
            pdf.cell(0, 10, "1. ANTECEDENTES GENERALES DEL PUESTO", 1, 1, 'L', 1)
            pdf.set_font("Arial", size=10)
            pdf.ln(5)
            
            # Tabla de datos
            pdf.set_fill_color(240, 240, 240)
            col_width = 45
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(col_width, 8, "Empresa:", 1, 0, 'L', 1)
            pdf.set_font("Arial", size=10)
            pdf.cell(col_width*1.5, 8, empresa, 1, 0, 'L')
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(col_width, 8, "RUT:", 1, 0, 'L', 1)
            pdf.set_font("Arial", size=10)
            pdf.cell(col_width, 8, rut, 1, 1, 'L')
            
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(col_width, 8, "Trabajador:", 1, 0, 'L', 1)
            pdf.set_font("Arial", size=10)
            pdf.cell(col_width*1.5, 8, nombre_trabajador, 1, 0, 'L')
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(col_width, 8, "Cargo:", 1, 0, 'L', 1)
            pdf.set_font("Arial", size=10)
            pdf.cell(col_width, 8, cargo, 1, 1, 'L')
            
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(col_width, 8, "Antigüedad:", 1, 0, 'L', 1)
            pdf.set_font("Arial", size=10)
            pdf.cell(col_width*1.5, 8, f"{antiguedad} años", 1, 0, 'L')
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(col_width, 8, "Ciclo:", 1, 0, 'L', 1)
            pdf.set_font("Arial", size=10)
            pdf.cell(col_width, 8, f"{ciclo_seg} seg", 1, 1, 'L')

            # Resultados IA
            pdf.ln(10)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_fill_color(200, 220, 255)
            pdf.cell(0, 10, "2. ANALISIS BIOMECANICO (MOTION TECH AI)", 1, 1, 'L', 1)
            pdf.ln(5)
            
            # Tabla de Resultados
            pdf.set_font("Arial", 'B', 10)
            pdf.set_fill_color(230, 230, 230)
            pdf.cell(60, 8, "Segmento", 1, 0, 'C', 1)
            pdf.cell(45, 8, "Ángulo Máximo", 1, 0, 'C', 1)
            pdf.cell(45, 8, "% Tiempo en Riesgo", 1, 0, 'C', 1)
            pdf.cell(40, 8, "Nivel de Riesgo", 1, 1, 'C', 1)
            
            pdf.set_font("Arial", size=10)
            
            # Fila Hombro
            pdf.cell(60, 8, "Hombro (Flexión)", 1, 0, 'L')
            pdf.cell(45, 8, f"{np.array(st.session_state['h_data']).max()}°", 1, 0, 'C')
            crit_h_val = int((np.array(st.session_state['h_data']) > 60).mean()*100)
            pdf.cell(45, 8, f"{crit_h_val}% (>60°)", 1, 0, 'C')
            
            riesgo_h = st.session_state['riesgo_h']
            if riesgo_h == "ALTO": pdf.set_text_color(200, 0, 0)
            elif riesgo_h == "MEDIO": pdf.set_text_color(200, 150, 0)
            else: pdf.set_text_color(0, 150, 0)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(40, 8, riesgo_h, 1, 1, 'C')
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", size=10)

            # Fila Tronco
            pdf.cell(60, 8, "Tronco (Flexión)", 1, 0, 'L')
            pdf.cell(45, 8, f"{np.array(st.session_state['t_data']).max()}°", 1, 0, 'C')
            crit_t_val = int((np.array(st.session_state['t_data']) > 60).mean()*100)
            pdf.cell(45, 8, f"{crit_t_val}% (>60°)", 1, 0, 'C')
            
            riesgo_t = st.session_state['riesgo_t']
            if riesgo_t == "ALTO": pdf.set_text_color(200, 0, 0)
            elif riesgo_t == "MEDIO": pdf.set_text_color(200, 150, 0)
            else: pdf.set_text_color(0, 150, 0)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(40, 8, riesgo_t, 1, 1, 'C')
            pdf.set_text_color(0, 0, 0)

            # Otras métricas
            pdf.ln(5)
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 8, f"Frecuencia de movimientos estimada: {int(60/ciclo_seg)} ciclos por minuto.", 0, 1)
            
            # --- SECCIÓN DE RECOMENDACIONES INTELIGENTES ---
            pdf.ln(10)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_fill_color(200, 220, 255)
            pdf.cell(0, 10, "3. CONCLUSIONES Y RECOMENDACIONES TÉCNICAS", 1, 1, 'L', 1)
            pdf.ln(5)
            
            pdf.set_font("Arial", size=11)
            for i, rec in enumerate(recoms):
                pdf.multi_cell(0, 8, f"{i+1}. {rec}", 0, 'L')
                pdf.ln(2)

            # Pie de página
            pdf.set_y(-30)
            pdf.set_font("Arial", 'I', 8)
            pdf.cell(0, 10, f"Documento generado automáticamente por Motion Tech AI el {time.strftime('%d/%m/%Y')}. Requiere validación de un ergónomo certificado.", 0, 0, 'C')

            output_file = f"EPT_{nombre_trabajador.replace(' ', '_')}.pdf"
            pdf.output(output_file)
            with open(output_file, "rb") as f:
                st.download_button("📥 DESCARGAR INFORME OFICIAL", f, file_name=output_file)
