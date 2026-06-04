# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate

# Configuración de la interfaz
st.set_page_config(layout="wide", page_title="Diagrama de Equilibrio Líquido-Vapor")

st.title("📊 Herramienta de Análisis de Diagramas de Fase en Equilibrio")
st.write("Esta aplicación interactiva permite ajustar curvas de equilibrio, calcular la regla de la palanca y analizar composiciones.")

# Inicializar estados de la sesión para líneas manuales
if 'manual_lines' not in st.session_state:
    st.session_state.manual_lines = []

# ==========================================
# 1. ENTRADA DE DATOS (PROCESAMIENTO POR ESPACIOS)
# ==========================================
st.sidebar.header("📋 1. Configuración de Datos")

# Pregunta si el eje Y es Temperatura o Presión
eje_y_tipo = st.sidebar.radio("Selecciona la variable del eje Y:", ["Temperatura", "Presión"])
unidad = "K" if eje_y_tipo == "Temperatura" else "atm/bar"

st.sidebar.subheader("Puntos de Datos de Equilibrio")
st.sidebar.write("Introduce tus datos experimentales separados por ESPACIOS:")

# Valores por defecto separados únicamente por espacios
default_x = "0.0 0.1 0.2 0.3 0.5 0.7 0.9 1.0"
default_y = "0.0 0.22 0.41 0.57 0.78 0.90 0.98 1.0"
default_tp = "373.15 365.0 358.5 353.0 345.0 340.0 338.5 337.65" if eje_y_tipo == "Temperatura" else "1.0 1.2 1.5 1.8 2.3 2.8 3.2 3.5"

x_input = st.sidebar.text_input("Fracción molar Líquido (x):", default_x)
y_input = st.sidebar.text_input("Fracción molar Vapor (y):", default_y)
tp_input = st.sidebar.text_input(f"{eje_y_tipo} ({unidad}):", default_tp)

def parse_inputs(input_string):
    clean_string = input_string.replace(",", " ")
    return np.array([float(i) for i in clean_string.split()])

# Procesamiento numérico
try:
    x_arr = parse_inputs(x_input)
    y_arr = parse_inputs(y_input)
    tp_arr = parse_inputs(tp_input)

    if not (len(x_arr) == len(y_arr) == len(tp_arr)):
        st.sidebar.error("⚠️ Error: Los vectores de x, y y T/P deben tener el mismo tamaño de elementos.")
        st.stop()
except ValueError:
    st.sidebar.error("⚠️ Error: Asegúrate de ingresar únicamente números válidos separados por espacios.")
    st.stop()

# Mostrar la tabla de datos procesados
df_datos = pd.DataFrame({'x (Líquido)': x_arr, 'y (Vapor)': y_arr, eje_y_tipo: tp_arr})
st.sidebar.dataframe(df_datos, use_container_width=True)

# ==========================================
# AJUSTE DE LA MEJOR CURVA POSIBLE (PCHIP Spline)
# ==========================================
sort_idx = np.argsort(tp_arr)
x_sorted, y_sorted, tp_sorted = x_arr[sort_idx], y_arr[sort_idx], tp_arr[sort_idx]

tp_smooth = np.linspace(tp_sorted.min(), tp_sorted.max(), 1000) 
x_smooth = pchip_interpolate(tp_sorted, x_sorted, tp_smooth)
y_smooth = pchip_interpolate(tp_sorted, y_sorted, tp_smooth)

# ==========================================
# 2. PARAMETROS DE ALIMENTACIÓN ESCRITOS POR EL USUARIO
# ==========================================
st.sidebar.header("⛽ 2. Parámetros de Alimentación")
z_feed = st.sidebar.number_input("Escribe la composición de alimentación (z):", min_value=0.0, max_value=1.0, value=0.2730, step=0.0001, format="%.4f")

st.sidebar.subheader("🕹️ Avanzar/Retroceder en la Regla")
control_regla = st.sidebar.radio("Controlar regla horizontal mediante:", [f"{eje_y_tipo}", "Fracción de Vapor Requerida"])

min_tp, max_tp = float(tp_sorted.min()), float(tp_sorted.max())
current_tp = (min_tp + max_tp) / 2

if control_regla == f"{eje_y_tipo}":
    current_tp = st.sidebar.slider(f"Mover regla vertical en {eje_y_tipo}:", min_tp, max_tp, current_tp, 0.05)
else:
    frac_vapor_req = st.sidebar.slider("Fracción de Vapor Requerida:", 0.0, 1.0, 0.60, 0.01)
    
    # REGLA DE LA PALANCA INVERSA CONTINUA
    with np.errstate(divide='ignore', invalid='ignore'):
        perfil_frac_v = (z_feed - x_smooth) / (y_smooth - x_smooth)
        zona_bifasica_idx = ((z_feed >= x_smooth) & (z_feed <= y_smooth)) | ((z_feed <= x_smooth) & (z_feed >= y_smooth))
        
    if np.any(zona_bifasica_idx):
        tp_bifasica = tp_smooth[zona_bifasica_idx]
        frac_bifasica = perfil_frac_v[zona_bifasica_idx]
        
        valid_mask = (~np.isnan(frac_bifasica)) & (frac_bifasica >= 0.0) & (frac_bifasica <= 1.0)
        if
