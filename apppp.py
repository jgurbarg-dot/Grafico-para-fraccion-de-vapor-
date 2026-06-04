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
        if np.any(valid_mask):
            tp_bifasica = tp_bifasica[valid_mask]
            frac_bifasica = frac_bifasica[valid_mask]
            
            sort_f_idx = np.argsort(frac_bifasica)
            current_tp = float(np.interp(frac_vapor_req, frac_bifasica[sort_f_idx], tp_bifasica[sort_f_idx]))
    else:
        current_tp = (min_tp + max_tp) / 2

# Evaluar composiciones exactas de equilibrio
x_eq = np.clip(pchip_interpolate(tp_sorted, x_sorted, current_tp), 0.0, 1.0)
y_eq = np.clip(pchip_interpolate(tp_sorted, y_sorted, current_tp), 0.0, 1.0)

x_izq, x_der = (x_eq, y_eq) if x_eq < y_eq else (y_eq, x_eq)

# ==========================================
# CÁLCULOS (LOGICA INVERSA)
# ==========================================
st.header("🧮 Cálculos en Tiempo Real")
col1, col2, col3, col4 = st.columns(4)

en_zona_bifasica = (x_izq <= z_feed <= x_der)

if en_zona_bifasica:
    segmento_izq = abs(z_feed - x_izq)
    segmento_der = abs(x_der - z_feed)
    
    # El segmento izquierdo (distancia al líquido) representa cuantitativamente al VAPOR
    # El segmento derecho (distancia al vapor) representa cuantitativamente al LÍQUIDO
    Distancia_V = segmento_izq  
    Distancia_L = segmento_der  
    
    frac_vapor_calculada = Distancia_V / (Distancia_L + Distancia_V) if (Distancia_L + Distancia_V) > 0 else 0.0

    col1.metric(f"{eje_y_tipo} Resultante", f"{current_tp:.2f} {unidad}")
    col2.metric("Distancia L (Prop. Líquido)", f"{Distancia_L:.4f}")
    col3.metric("Distancia V (Prop. Vapor)", f"{Distancia_V:.4f}")
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

# Curvas continuas suaves
ax.plot(x_smooth, tp_smooth, color='red', label='Línea de Líquido Saturado (x)', linewidth=2)
ax.plot(y_smooth, tp_smooth, color='blue', linestyle='--', label='Línea de Vapor Saturado (y)', linewidth=2)

# Puntos muestreales originales
ax.scatter(x_arr, tp_arr, color='darkred', marker='x', s=60, label='Puntos datos Líquido')
ax.scatter(y_arr, tp_arr, color='darkblue', marker='o', s=50, label='Puntos datos Vapor')

# Alimentación vertical fija escrita por el usuario
ax.axvline(x=z_feed, color='#0f2c59', linewidth=2.5, label=f'Alimentación (z = {z_feed:.4f})')

# Trazado físico de la recta de reparto
if en_zona_bifasica:
    ax.plot([x_izq, x_der], [current_tp, current_tp], color='black', linewidth=2.5, marker='|', markersize=12)
    
    # CORRECCIÓN EN GRÁFICA:
    # El segmento izquierdo (entre x_izq y z_feed) representa la proporción de VAPOR -> Va la letra 'V'
    # El segmento derecho (entre z_feed y x_der) representa la proporción de LÍQUIDO -> Va la letra 'L'
    ax.text((x_izq + z_feed)/2, current_tp + (max_tp-min_tp)*0.015, 'V', fontsize=12, weight='bold', ha='center', color='blue')
    ax.text((z_feed + x_der)/2, current_tp + (max_tp-min_tp)*0.015, 'L', fontsize=12, weight='bold', ha='center', color='red')

# Nodo de operación central (+ negro)
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

# PUNTERO INTERACTIVO
st.sidebar.header("🎯 4. Puntero Equivalente")
pos_puntero = st.sidebar.slider("Mover Puntero sobre Curvas:", min_tp, max_tp, current_tp)
x_p = pchip_interpolate(tp_sorted, x_sorted, pos_puntero)
y_p = pchip_interpolate(tp_sorted, y_sorted, pos_puntero)

ax.plot([x_p, y_p], [pos_puntero, pos_puntero], color='green', linestyle=':', alpha=0.8)
ax.scatter([x_p, y_p], [pos_puntero, pos_puntero], color='green', marker='s', s=40)

st.sidebar.info(f"📍 **Equivalencias del Puntero:**\n- Líquido Equivalente (x): {x_p:.4f}\n- Vapor Equivalente (y): {y_p:.4f}")

# Rejilla y límites del gráfico
ax.set_xlabel("Fracción Molar (x, y)", fontsize=11)
ax.set_ylabel(f"{eje_y_tipo} ({unidad})", fontsize=11)
ax.set_xlim(0, 1)
ax.set_xticks(np.arange(0, 1.05, 0.05))
ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgray')
ax.legend(loc='best')
plt.tight_layout()

st.pyplot(fig)
