import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Cargar dataset
df = pd.read_csv("datos/merged_asteroid_data.csv")
print(df.describe())

# Histograma de probabilidad de impacto
df["Cumulative Impact Probability"].hist(bins=10)
plt.title("Distribución de probabilidad de impacto")
plt.xlabel("Probabilidad acumulada")
plt.ylabel("Frecuencia")
plt.show()


# Etiquetado binario: impacto probable si probabilidad > 1e-5
df["impacto_probable"] = df["Cumulative Impact Probability"].apply(lambda x: 1 if x > 1e-5 else 0)

# Columnas predictoras
columnas_predictoras = [
    "Asteroid Velocity",
    "Asteroid Diameter (km)",
    "Maximum Palermo Scale",
    "Maximum Torino Scale",
    "Asteroid Magnitude_orbit",
    "Orbit Axis (AU)",
    "Orbit Eccentricity",
    "Orbit Inclination (deg)"
]

# Eliminar filas con valores no numéricos o nulos en las columnas predictoras
df_clean = df[columnas_predictoras + ["impacto_probable"]].apply(pd.to_numeric, errors="coerce")
df_clean["impacto_probable"] = df["impacto_probable"]
df_clean = df_clean.dropna()

# Variables X e y
X = df_clean[columnas_predictoras].values
y = df_clean["impacto_probable"].values


# División estratificada de datos
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y,
                                                                    test_size=0.3,
                                                                    random_state=0,
                                                                    stratify=y)

# Regresión logística
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_clf, y_train_clf)
y_pred_logreg_train = logreg.predict(X_train_clf)
y_pred_logreg_test = logreg.predict(X_test_clf)

acc_logreg_train = accuracy_score(y_train_clf, y_pred_logreg_train)
acc_logreg_test = accuracy_score(y_test_clf, y_pred_logreg_test)

print("\nResultados regresión logística")
print(f"Acc (train): {acc_logreg_train:.3f}")
print(f"Acc (test): {acc_logreg_test:.3f}")
print(classification_report(y_test_clf, y_pred_logreg_test, labels=[0, 1], target_names=["no probable", "probable"]))



# k-NN con varios valores de k
ks = [1, 3, 5, 7, 9, 11]
metricas = []

for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_clf, y_train_clf)
    y_pred_val = knn.predict(X_test_clf)
    acc = accuracy_score(y_test_clf, y_pred_val)
    metricas.append(acc)

# Gráfica de exactitud k
fig, ax = plt.subplots()
ax.plot(ks, metricas, marker="o")
ax.set_xlabel("Número de vecinos (k)")
ax.set_ylabel("Exactitud")
ax.set_title("Exactitud de k-NN para diferentes k")
plt.show()

# Mejor k
best_idx = np.argmax(metricas)
best_k = ks[best_idx]
print(f"\nMejor valor de k: {best_k} con exactitud {metricas[best_idx]:.3f}")

# Reentrenar con mejor k
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_clf, y_train_clf)
y_pred_knn = best_knn.predict(X_test_clf)


cm_best_knn = confusion_matrix(y_test_clf, y_pred_knn)
disp_best_knn = ConfusionMatrixDisplay(confusion_matrix=cm_best_knn, display_labels=["No probable", "Probable"])
disp_best_knn.plot(cmap="Blues")
plt.title(f"Matriz de confusión - k-NN (k={best_k})")
plt.show()


print("\nResultados KNN con mejor k")
print(classification_report(y_test_clf, y_pred_knn, labels=[0, 1], target_names=["no probable", "probable"]))


# Ablation: quitar 'Asteroid Velocity'
columnas_sin_velocidad = [col for col in columnas_predictoras if col != "Asteroid Velocity"]
X_ablation = df_clean[columnas_sin_velocidad].values

X_train_ab, X_test_ab, y_train_ab, y_test_ab = train_test_split(X_ablation, y,
                                                                test_size=0.3,
                                                                random_state=0,
                                                                stratify=y)

# Regresión logística sin velocidad
pipeline_logreg_ab = Pipeline([("scaler", StandardScaler()),
                               ("model", LogisticRegression(max_iter=1000))])
pipeline_logreg_ab.fit(X_train_ab, y_train_ab)
y_pred_logreg_ab = pipeline_logreg_ab.predict(X_test_ab)

acc_logreg_ab = accuracy_score(y_test_ab, y_pred_logreg_ab)
cm_logreg_ab = confusion_matrix(y_test_ab, y_pred_logreg_ab)

print("\nResultados regresión logística (sin velocidad)")
print(f"Accuracy: {acc_logreg_ab:.3f}")
disp_logreg_ab = ConfusionMatrixDisplay(confusion_matrix=cm_logreg_ab, display_labels=["No probable", "Probable"])
disp_logreg_ab.plot(cmap="Blues")
plt.title("Matriz de confusión - Regresión logística (sin velocidad)")
plt.show()

# k-NN sin velocidad
pipeline_knn_ab = Pipeline([("scaler", StandardScaler()),
                            ("model", KNeighborsClassifier(n_neighbors=best_k))])
pipeline_knn_ab.fit(X_train_ab, y_train_ab)
y_pred_knn_ab = pipeline_knn_ab.predict(X_test_ab)

acc_knn_ab = accuracy_score(y_test_ab, y_pred_knn_ab)
cm_knn_ab = confusion_matrix(y_test_ab, y_pred_knn_ab)

print(f"\nResultados k-NN (k={best_k}) sin velocidad")
print(f"Accuracy: {acc_knn_ab:.3f}")
disp_knn_ab = ConfusionMatrixDisplay(confusion_matrix=cm_knn_ab, display_labels=["No probable", "Probable"])
disp_knn_ab.plot(cmap="Blues")
plt.title(f"Matriz de confusión - k-NN (k={best_k}) sin velocidad")
plt.show()