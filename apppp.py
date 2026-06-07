# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate

# Configuración de la interfaz
st.set_page_config(layout="wide", page_title="Diagrama de Equilibrio Líquido-Vapor")

st.title("📊 Herramienta de Análisis de Diagramas de Fase en Equilibrio (Azeótropo Soportado)")
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
# NUEVO AJUSTE: INTERPOLACIÓN RESPECTO A COMPOSICIONES (X e Y)
# ==========================================
# Como x e y son crecientes, ordenamos cada curva de manera independiente respecto a su fracción molar
sort_x = np.argsort(x_arr)
x_sorted_for_tp, tp_sorted_for_x = x_arr[sort_x], tp_arr[sort_x]

sort_y = np.argsort(y_arr)
y_sorted_for_tp, tp_sorted_for_y = y_arr[sort_y], tp_arr[sort_y]

# Creamos un mallado continuo fino para el eje de composiciones (0 a 1)
comp_smooth = np.linspace(0.0, 1.0, 1000)

# Calculamos las temperaturas/presiones suaves en función de las composiciones
tp_smooth_x = pchip_interpolate(x_sorted_for_tp, tp_sorted_for_x, comp_smooth) # Líquido saturado
tp_smooth_y = pchip_interpolate(y_sorted_for_tp, tp_sorted_for_y, comp_smooth) # Vapor saturado

# Rangos de T/P globales de los datos experimentales
min_tp, max_tp = float(tp_arr.min()), float(tp_arr.max())
min_smooth, max_smooth = min(tp_smooth_x.min(), tp_smooth_y.min()), max(tp_smooth_x.max(), tp_smooth_y.max())

# ==========================================
# 2. PARAMETROS DE ALIMENTACIÓN ESCRITOS POR EL USUARIO
# ==========================================
st.sidebar.header("⛽ 2. Parámetros de Alimentación")
z_feed = st.sidebar.number_input("Escribe la composición de alimentación (z):", min_value=0.0, max_value=1.0, value=0.2730, step=0.0001, format="%.4f")

st.sidebar.subheader("🕹️ Avanzar/Retroceder en la Regla")
control_regla = st.sidebar.radio("Controlar regla horizontal mediante:", [f"{eje_y_tipo}", "Fracción de Vapor Requerida"])

current_tp = (min_tp + max_tp) / 2

if control_regla == f"{eje_y_tipo}":
    current_tp = st.sidebar.slider(f"Mover regla vertical en {eje_y_tipo}:", float(min_smooth), float(max_smooth), float(current_tp), 0.05)
    
    # Encontrar todas las intersecciones de la T/P actual con las curvas de líquido (x) y vapor (y)
    # Al haber un azeótropo, puede haber múltiples cruces. Buscamos el más cercano o resolvemos por interpolación local.
    x_eq = np.interp(current_tp, tp_smooth_x, comp_smooth, left=np.nan, right=np.nan)
    y_eq = np.interp(current_tp, tp_smooth_y, comp_smooth, left=np.nan, right=np.nan)
    
    # Si falla la interpolación simple por la inversión del azeótropo, usamos aproximación por máscaras
    if np.isnan(x_eq):
        idx_x = np.argmin(np.abs(tp_smooth_x - current_tp))
        x_eq = comp_smooth[idx_x]
    if np.isnan(y_eq):
        idx_y = np.argmin(np.abs(tp_smooth_y - current_tp))
        y_eq = comp_smooth[idx_y]
else:
    frac_vapor_req = st.sidebar.slider("Fracción de Vapor Requerida:", 0.0, 1.0, 0.60, 0.01)
    
    # REGLA DE LA PALANCA INVERSA CONTINUA ADAPTADA
    with np.errstate(divide='ignore', invalid='ignore'):
        perfil_frac_v = (z_feed - comp_smooth) / (comp_smooth - comp_smooth) # dummy estructural
        
        # En cada punto 'comp_smooth' asumido como x, calculamos su T y su y correspondiente
        tp_hipotetica = tp_smooth_x
        # Buscamos qué 'y' tiene esa misma temperatura en la otra curva
        y_correspondiente = np.array([comp_smooth[np.argmin(np.abs(tp_smooth_y - t))] for t in tp_hipotetica])
        
        frac_bifasica = (z_feed - comp_smooth) / (y_correspondiente - comp_smooth)
        zona_bifasica_idx = ((z_feed >= comp_smooth) & (z_feed <= y_correspondiente)) | ((z_feed <= comp_smooth) & (z_feed >= y_correspondiente))
        
    if np.any(zona_bifasica_idx):
        tp_bifasica = tp_hipotetica[zona_bifasica_idx]
        frac_bifasica = frac_bifasica[zona_bifasica_idx]
        
        valid_mask = (~np.isnan(frac_bifasica)) & (frac_bifasica >= 0.0) & (frac_bifasica <= 1.0)
        if np.any(valid_mask):
            tp_bifasica = tp_bifasica[valid_mask]
            frac_bifasica = frac_bifasica[valid_mask]
            
            sort_f_idx = np.argsort(frac_bifasica)
            current_tp = float(np.interp(frac_vapor_req, frac_bifasica[sort_f_idx], tp_bifasica[sort_f_idx]))
        else:
            current_tp = (min_tp + max_tp) / 2
    else:
        current_tp = (min_tp + max_tp) / 2

    # Recalcular x_eq e y_eq precisos para la T/P obtenida
    idx_x = np.argmin(np.abs(tp_smooth_x - current_tp))
    x_eq = comp_smooth[idx_x]
    idx_y = np.argmin(np.abs(tp_smooth_y - current_tp))
    y_eq = comp_smooth[idx_y]

x_eq = np.clip(x_eq, 0.0, 1.0)
y_eq = np.clip(y_eq, 0.0, 1.0)
x_izq, x_der = (x_eq, y_eq) if x_eq < y_eq else (y_eq, x_eq)

# ==========================================
# CÁLCULOS (LOGICA INVERSA CON LEYENDAS INTERCAMBIADAS)
# ==========================================
st.header("🧮 Cálculos en Tiempo Real")
col1, col2, col3, col4 = st.columns(4)

en_zona_bifasica = (x_izq <= z_feed <= x_der)

if en_zona_bifasica:
    segmento_izq = abs(z_feed - x_izq)
    segmento_der = abs(x_der - z_feed)
    
    Distancia_L = segmento_izq  
    Distancia_V = segmento_der  
    
    frac_vapor_calculada = Distancia_L / (Distancia_L + Distancia_V) if (Distancia_L + Distancia_V) > 0 else 0.0

    col1.metric(f"{eje_y_tipo} Resultante", f"{current_tp:.2f} {unidad}")
    col2.metric("Distancia L", f"{Distancia_L:.4f}")
    col3.metric("Distancia V", f"{Distancia_V:.4f}")
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

# Curvas continuas suaves (Graficamos X en el eje horizontal, T/P calculada en el eje vertical)
ax.plot(comp_smooth, tp_smooth_x, color='red', label='Línea de Líquido Saturado (x)', linewidth=2)
ax.plot(comp_smooth, tp_smooth_y, color='blue', linestyle='--', label='Línea de Vapor Saturado (y)', linewidth=2)

# Puntos muestreales originales
ax.scatter(x_arr, tp_arr, color='darkred', marker='x', s=60, label='Puntos datos Líquido')
ax.scatter(y_arr, tp_arr, color='darkblue', marker='o', s=50, label='Puntos datos Vapor')

# Alimentación vertical fija escrita por el usuario
ax.axvline(x=z_feed, color='#0f2c59', linewidth=2.5, label=f'Alimentación (z = {z_feed:.4f})')

# Trazado físico de la recta de reparto
if en_zona_bifasica:
    ax.plot([x_izq, x_der], [current_tp, current_tp], color='black', linewidth=2.5, marker='|', markersize=12)
    
    pos_L_grafica = x_izq + (x_der - x_izq) * 0.03
    pos_V_grafica = x_der - (x_der - x_izq) * 0.03
    
    ax.text(pos_L_grafica, current_tp + (max_tp-min_tp)*0.015, 'L', fontsize=12, weight='bold', ha='left', color='red')
    ax.text(pos_V_grafica, current_tp + (max_tp-min_tp)*0.015, 'V', fontsize=12, weight='bold', ha='right', color='blue')

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

# PUNTERO INTERACTIVO CORREGIDO PARA AZEÓTROPOS
st.sidebar.header("🎯 4. Puntero Equivalente")

# Buscamos las posiciones asociadas al slider utilizando las curvas dependientes de la composición
pos_puntero_comp = st.sidebar.slider("Mover Puntero sobre Composiciones (X/Y):", 0.0, 1.0, 0.5, 0.01)
tp_p_x = pchip_interpolate(x_sorted_for_tp, tp_sorted_for_x, pos_puntero_comp)
tp_p_y = pchip_interpolate(y_sorted_for_tp, tp_sorted_for_y, pos_puntero_comp)

# Marcadores verdes del puntero
ax.scatter([pos_puntero_comp], [tp_p_x], color='green', marker='s', s=40)
ax.scatter([pos_puntero_comp], [tp_p_y], color='green', marker='s', s=40)

st.sidebar.info(f"📍 **Equivalencias del Puntero (para Z = {pos_puntero_comp:.2f}):**\n- T/P Líquido (x): {tp_p_x:.2f} {unidad}\n- T/P Vapor (y): {tp_p_y:.2f} {unidad}")

# Rejilla y límites del gráfico
ax.set_xlabel("Fracción Molar (x, y)", fontsize=11)
ax.set_ylabel(f"{eje_y_tipo} ({unidad})", fontsize=11)
ax.set_xlim(0, 1)
ax.set_xticks(np.arange(0, 1.05, 0.05))
ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgray')
ax.legend(loc='best')
plt.tight_layout()

st.pyplot(fig)
