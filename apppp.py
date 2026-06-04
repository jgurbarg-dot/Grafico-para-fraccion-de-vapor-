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
default_h = "100 120 145 170 210 260 310 350"  # Entalpía por defecto (kJ/kg o kcal/mol)

x_input = st.sidebar.text_input("Fracción molar Líquido (x):", default_x)
y_input = st.sidebar.text_input("Fracción molar Vapor (y):", default_y)
tp_input = st.sidebar.text_input(f"{eje_y_tipo} ({unidad}):", default_tp)
h_input = st.sidebar.text_input("Entalpia (H):", default_h)

# Función auxiliar para parsear texto separado por espacios o comas
def parse_inputs(input_string):
    # Reemplaza comas por espacios por si acaso el usuario mezcla formatos, luego separa
    clean_string = input_string.replace(",", " ")
    return np.array([float(i) for i in clean_string.split()])

# Procesamiento y conversión de datos de texto a arreglos numéricos
try:
    x_arr = parse_inputs(x_input)
    y_arr = parse_inputs(y_input)
    tp_arr = parse_inputs(tp_input)
    h_arr = parse_inputs(h_input)

    if not (len(x_arr) == len(y_arr) == len(tp_arr) == len(h_arr)):
        st.sidebar.error("⚠️ Error: Los vectores de x, y, T/P y Entalpía deben tener el mismo tamaño de elementos.")
        st.stop()
except ValueError:
    st.sidebar.error("⚠️ Error: Asegúrate de ingresar únicamente números válidos separados por espacios.")
    st.stop()

# Mostrar la tabla de datos procesados incluyendo Entalpía
df_datos = pd.DataFrame({'x (Líquido)': x_arr, 'y (Vapor)': y_arr, eje_y_tipo: tp_arr, 'Entalpía (H)': h_arr})
st.sidebar.dataframe(df_datos, use_container_width=True)

# ==========================================
# AJUSTE DE LA MEJOR CURVA POSIBLE (PCHIP Spline)
# ==========================================
sort_idx = np.argsort(tp_arr)
x_sorted, y_sorted, tp_sorted = x_arr[sort_idx], y_arr[sort_idx], tp_arr[sort_idx]

# Generación de curvas suaves de alta resolución
tp_smooth = np.linspace(tp_sorted.min(), tp_sorted.max(), 500)
x_smooth = pchip_interpolate(tp_sorted, x_sorted, tp_smooth)
y_smooth = pchip_interpolate(tp_sorted, y_sorted, tp_smooth)

# ==========================================
# 2. PARAMETROS DE ALIMENTACIÓN ESCRITOS POR EL USUARIO
# ==========================================
st.sidebar.header("⛽ 2. Parámetros de Alimentación")

# Ahora la alimentación es un dato a ESCRIBIR numéricamente por el usuario
z_feed = st.sidebar.number_input("Escribe la composición de alimentación (z):", min_value=0.0, max_value=1.0, value=0.265, step=0.001, format="%.4f")

st.sidebar.subheader("🕹️ Avanzar/Retroceder en la Regla")

# Opción para decidir bajo qué parámetro queremos controlar la regla horizontal
control_regla = st.sidebar.radio("Controlar regla horizontal mediante:", [f"{eje_y_tipo}", "Fracción de Vapor Requerida"])

min_tp, max_tp = float(tp_sorted.min()), float(tp_sorted.max())

# Inicialización de variables clave
current_tp = (min_tp + max_tp) / 2
frac_vapor_req = 0.5

if control_regla == f"{eje_y_tipo}":
    # Mover por temperatura/presión (Como estaba originalmente)
    current_tp = st.sidebar.slider(f"Mover regla vertical en {eje_y_tipo}:", min_tp, max_tp, current_tp, 0.05)
else:
    # Mover por Fracción de Vapor Requerida
    frac_vapor_req = st.sidebar.slider("Fracción de Vapor Requerida:", 0.0, 1.0, 0.5, 0.01)
    
    # Acoplamiento numérico inverso: Buscar la T/P que genera dicha fracción de vapor
    # Calculamos el perfil de fracciones de vapor teóricas a lo largo de toda la curva de mezcla
    with np.errstate(divide='ignore', invalid='ignore'):
        # Encontrar extremos geométricos para cada punto del mallado suave
        left_s = np.minimum(x_smooth, y_smooth)
        right_s = np.maximum(x_smooth, y_smooth)
        
        # Distancias relativas en el mallado para simular la regla de la palanca continua
        L_s = np.abs(z_feed - left_s)
        V_s = np.abs(right_s - z_feed)
        perfil_frac_v = V_s / (V_s + L_s)
        
        # Filtrar solo la zona bifásica válida para la alimentación z dada
        zona_bifasica_idx = (z_feed >= left_s) & (z_feed <= right_s)
        
    if np.any(zona_bifasica_idx):
        tp_bifasica = tp_smooth[zona_bifasica_idx]
        frac_bifasica = perfil_frac_v[zona_bifasica_idx]
        
        # Interpolar para hallar la T/P exacta que corresponde a la fracción de vapor deseada
        # Ordenamos los vectores para asegurar una interpolación monótona limpia
        sort_f_idx = np.argsort(frac_bifasica)
        current_tp = float(np.interp(frac_vapor_req, frac_bifasica[sort_f_idx], tp_bifasica[sort_f_idx]))
    else:
        current_tp = (min_tp + max_tp) / 2

# Calcular las intersecciones exactas de las fases a la T/P calculada/seleccionada
x_c = np.clip(pchip_interpolate(tp_sorted, x_sorted, current_tp), 0.0, 1.0)
y_c = np.clip(pchip_interpolate(tp_sorted, y_sorted, current_tp), 0.0, 1.0)

if x_c > y_c:
    left_curve, right_curve = y_c, x_c
else:
    left_curve, right_curve = x_c, y_c

# ==========================================
# CÁLCULOS DE LA REGLA DE LA PALANCA EN TIEMPO REAL
# ==========================================
st.header("🧮 Cálculos en Tiempo Real")
col1, col2, col3, col4 = st.columns(4)

if left_curve <= z_feed <= right_curve:
    L_dist = abs(z_feed - left_curve)
    V_dist = abs(right_curve - z_feed)
    frac_vapor_calculada = V_dist / (V_dist + L_dist) if (V_dist + L_dist) > 0 else 0.0

    col1.metric(f"{eje_y_tipo} Resultante", f"{current_tp:.2f} {unidad}")
    col2.metric("Distancia L (Segmento Izq.)", f"{L_dist:.4f}")
    col3.metric("Distancia V (Segmento Der.)", f"{V_dist:.4f}")
    col4.metric("Fracción de Vapor V/(V+L)", f"{frac_vapor_calculada:.4f}")
else:
    col1.metric(f"{eje_y_tipo} Resultante", f"{current_tp:.2f} {unidad}")
    col2.write("Fuera de la región Bifásica")
    col3.write("")
    col4.metric("Fracción de Vapor", "0.0 o 1.0 (Monofásico)")

# ==========================================
# 3. GRAFICACIÓN DINÁMICA
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

# Graficar las curvas ajustadas suaves
ax.plot(x_smooth, tp_smooth, color='red', label='Línea de Líquido Saturado', linewidth=2)
ax.plot(y_smooth, tp_smooth, color='blue', linestyle='--', label='Línea de Vapor Saturado', linewidth=2)

# Graficar los puntos originales provistos por el usuario
ax.scatter(x_arr, tp_arr, color='darkred', marker='x', s=60, label='Puntos datos Líquido')
ax.scatter(y_arr, tp_arr, color='darkblue', marker='o', s=50, label='Puntos datos Vapor')

# Línea vertical de Alimentación basada en el dato escrito por el usuario
ax.axvline(x=z_feed, color='#0f2c59', linewidth=2.5, label=f'Alimentación (z = {z_feed:.4f})')

# Dibujar la regla horizontal móvil (Tie-line)
if left_curve <= z_feed <= right_curve:
    ax.plot([left_curve, right_curve], [current_tp, current_tp], color='black', linewidth=2.5, marker='|', markersize=12)
    ax.text((left_curve + z_feed)/2, current_tp + (max_tp-min_tp)*0.015, 'L', fontsize=12, weight='bold', ha='center')
    ax.text((z_feed + right_curve)/2, current_tp + (max_tp-min_tp)*0.015, 'V', fontsize=12, weight='bold', ha='center')

# Marcador sobre el eje central de alimentación
ax.scatter([z_feed], [current_tp], color='black', marker='+', s=150, zorder=5)

# ==========================================
# CONTROLES EXTRA: LÍNEAS MANUALES Y PUNTERO
# ==========================================
st.sidebar.header("🛠️ 3. Añadir Líneas Manuales")
tipo_linea = st.sidebar.selectbox("Tipo:", ["Horizontal", "Vertical"])
valor_linea = st.sidebar.number_input("Valor de la coordenada:", min_value=0.0, max_value=500.0, value=0.5, step=0.05)

if st.sidebar.button("➕ Añadir Línea"):
    st.session_state.manual_lines.append({'tipo': tipo_linea, 'valor': valor_linea})
if st.sidebar.button("🗑️ Borrar Líneas"):
    st.session_state.manual_lines = []

for line in st.session_state.manual_lines:
    if line['tipo'] == "Horizontal":
        ax.axhline(y=line['valor'], color='gray', linestyle=':', alpha=0.7)
    else:
        ax.axvline(x=line['valor'], color='gray', linestyle=':', alpha=0.7)

# PUNTERO INTERACTIVO AUTOMÁTICO
st.sidebar.header("🎯 4. Puntero Equivalente")
pos_puntero = st.sidebar.slider("Mover Puntero sobre Curvas:", min_tp, max_tp, current_tp)
x_p = pchip_interpolate(tp_sorted, x_sorted, pos_puntero)
y_p = pchip_interpolate(tp_sorted, y_sorted, pos_puntero)

ax.plot([x_p, y_p], [pos_puntero, pos_puntero], color='green', linestyle=':', alpha=0.8)
ax.scatter([x_p, y_p], [pos_puntero, pos_puntero], color='green', marker='s', s=40)

st.sidebar.info(f"📍 **Equivalencias del Puntero:**\n- Líquido Equivalente (x): {x_p:.4f}\n- Vapor Equivalente (y): {y_p:.4f}")

# Ajustes estéticos finales
ax.set_xlabel("Fracción Molar (x, y)", fontsize=11)
ax.set_ylabel(f"{eje_y_tipo} ({unidad})", fontsize=11)
ax.set_xlim(0, 1)
ax.set_xticks(np.arange(0, 1.05, 0.05))
ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgray')
ax.legend(loc='best')
plt.tight_layout()

st.pyplot(fig)
