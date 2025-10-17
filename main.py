import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# ===========================
# 1️⃣ Crear el DataFrame
# ===========================
df = pd.DataFrame({
    "Area": [2600, 3000, 3200, 3600, 4000],
    "Price": [550000, 565000, 610000, 680000, 725000]
})

# Dibujar los puntos reales (datos de entrenamiento)
plt.scatter(df.Area, df.Price, color="red", label="Datos reales")
plt.xlabel("Área (m²)")
plt.ylabel("Precio (Euros)")
plt.title("Regresión Lineal: Precio vs Área")

# ===========================
# 2️⃣ Preparar los datos
# ===========================
X = df[["Area"]]  # Variable independiente
y = df["Price"]   # Variable dependiente

# ===========================
# 3️⃣ Crear y entrenar el modelo
# ===========================
reg = linear_model.LinearRegression()
reg.fit(X, y)

# ===========================
# 4️⃣ Dibujar la línea de regresión (modelo)
# ===========================
y_pred_line = reg.predict(X)
plt.plot(df["Area"], y_pred_line, color="blue",
         linewidth=2, label="Línea de regresión")

# ===========================
# 5️⃣ Hacer una predicción concreta (2900 ft²)
# ===========================
area_pred = 2900
price_pred = reg.predict([[area_pred]])
print(
    f"El precio de una casa con {area_pred} m² es: {price_pred[0]:,.0f} Euros")

# ===========================
# 6️⃣ Añadir ese punto a la gráfica
# ===========================
# Dibujar el punto de predicción con su valor en el label
plt.scatter(
    area_pred,
    price_pred,
    color="green",
    s=100,
    marker="o",
    label=f"Predicción ({area_pred} m² → {price_pred.item():,.0f} €)"
)

# Etiquetas, leyenda y cuadrícula
plt.legend()
plt.grid(True)
plt.show(block=True)
