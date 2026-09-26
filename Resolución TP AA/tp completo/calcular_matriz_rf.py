"""Reproduce la matriz de test del bosque elegido en la búsqueda original.

Desde la raíz del repo: python "Resolución TP AA/tp completo/calcular_matriz_rf.py"
Requiere haber reconstruido el CSV mediante 00_base_de_datos/reconstruir_dataset.py.
"""
from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RES))
from _comun import cargar, pre  # noqa: E402

df = cargar()
X, y = df.drop(columns="target"), df["target"]
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
forest = Pipeline([
    ("pre", pre(Xtr, False)),
    ("modelo", RandomForestClassifier(n_estimators=500, max_depth=None,
                                       min_samples_leaf=1, max_features="sqrt",
                                       random_state=42, n_jobs=4)),
])
forest.fit(Xtr, ytr)
pred = forest.predict(Xte)
labels = ["Desertor", "En Curso", "Graduado"]
cm = confusion_matrix(yte, pred, labels=labels)
rows = [{"real": real, "predicha": predicted, "n": int(cm[i, j]),
         "n_clase": int(cm[i].sum()), "porcentaje_fila": 100*cm[i, j]/cm[i].sum()}
        for i, real in enumerate(labels) for j, predicted in enumerate(labels)]
out = Path(__file__).with_name("matriz_rf.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(pd.DataFrame(cm, index=labels, columns=labels).to_string())
print(f"F1 macro={f1_score(yte,pred,average='macro'):.4f}")
print("recall:", dict(zip(labels, recall_score(yte,pred,labels=labels,average=None).round(4))))
print(out)
