# Del material de la cátedra a esta alternativa

Los seis Colab están en la carpeta privada de Drive `AA`:
https://drive.google.com/drive/folders/1vDhEWdZ6TvOOlihiFv--cPeWxm4iTUB5

| Cátedra | Ejercicio o idea | Aplicación aquí |
|---|---|---|
| [Clase 1: EDA](https://colab.research.google.com/drive/1PQ2aDg5KFQ6fKOUMaJPkvyowtV8FRybC) | Tipos, faltantes, distribuciones y relaciones con la respuesta. | EDA sobre train; tasas condicionales con denominadores, ceros y positivos por separado, categorías pequeñas y redundancia. |
| [Clase 2: conceptos](https://colab.research.google.com/drive/1eDlAVHOb16r-DY2FK5u-l9sb_6XoeQEZ) | Regla de corte, TP/TN/FP/FN y métricas. | Matriz binaria; se distingue «hipótesis» como regla de clasificación de las hipótesis sustantivas H1–H5. |
| [Clase 3: selección](https://colab.research.google.com/drive/14A02eD4ScxwzBujn8m3f7ta5sv7wUCcC) | Test apartado, CV dentro de desarrollo, curva de complejidad. | Split 80/20 estratificado, mismos folds entre etapas, CV de 3 folds solo en train. |
| [Clase 4: regresión](https://colab.research.google.com/drive/1X_VtjLkt0xQWRbdarhzKPBRPy5ONCFzD) | Regularización y colinealidad en regresión lineal. | La regularización de Logística es una analogía; no se trasladan sus coeficientes ni resultados numéricos al target. |
| [Clase 5: clasificación](https://colab.research.google.com/drive/1mqjDIpQInetyRm4v6-9san9UmN5UEHU-) | Binario/multiclase, métricas por clase, selección dentro de train. | ROC-AUC y AP binarios; F1 macro, F1 por clase y matrices multiclase. |
| [Clase 6: árboles](https://colab.research.google.com/drive/1HsFNINe9GdZZBXCyzC1f9SETkv8hzHEZ) | Profundidad, sobreajuste, RandomizedSearchCV y Dummy. | Árbol multiclase con búsqueda acotada, CV y baseline Dummy. |

## Decisiones que no vienen «dadas» por los Colab

- El mismo algoritmo compara etapas dentro de cada tarea: Regresión Logística
  en binario y Árbol en multiclase. **No** se interpreta el contraste entre
  métricas de estas dos tareas como efecto de cambiar algoritmo.
- Para esta base de 50.000 filas usamos `GridSearchCV` de 3 folds, grillas
  explícitas y pipelines; la clase 6 ejemplifica `RandomizedSearchCV` de 5
  folds en datos simulados. Ninguna grilla demuestra un óptimo universal.
- La muestra de evaluación ya se observó en el TP anterior. Por ello esta
  alternativa es **exploratoria** aun cuando las relaciones con el target
  que originan hipótesis se describen aquí solo con train.
- En la clase 3 hay un ejemplo didáctico que llama CV sobre `X,y` completo;
  seguimos su regla escrita de dejar el test fuera del desarrollo. En la
  clase 6 se grafican profundidades contra test para enseñar sobreajuste;
  aquí no se selecciona profundidad con test.
