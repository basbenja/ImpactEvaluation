# Framework de Simulación: Programa de Crédito Productivo

## Panel de Datos para Evaluación de Impacto

---

### Objetivo

Este notebook implementa un framework de simulación para generar datos de panel sintéticos que replican las características de un programa de crédito productivo para PyMEs. El objetivo es disponer de un banco de datos donde **conocemos el verdadero efecto causal**, permitiendo:

1. Evaluar el desempeño de diferentes estimadores (DiD, TWFE, Event Study, etc.)
2. Cuantificar el sesgo de selección
3. Ilustrar conceptos de identificación causal

### Características del modelo

| Característica | Descripción |
|----------------|-------------|
| Estructura | Panel: $N$ empresas × $T$ períodos |
| Tratamiento | Escalonado (staggered) en $K$ cohortes |
| Selección | Sesgo parametrizable vía variables no observables |
| Efectos | Dinámicos (crecen post-tratamiento) y heterogéneos |
| Outcomes | Empleados (entero), Salario (positivo), Crédito (binario) |

---

## 1. CONFIGURACIÓN DEL MODELO

La configuración centraliza **todos los parámetros** del modelo en un único diccionario. Esto permite:
- Modificar el DGP sin tocar el código
- Realizar análisis de sensibilidad
- Documentar explícitamente los supuestos

---

### 1.1 Parámetros Generales y Estructura Temporal

```python
'n_empresas': 2000,              # Tamaño de la muestra
'n_periodos': 12,                # Horizonte temporal
'periodo_inicio_programa': 5,    # Cuándo comienza el programa
'frecuencia': 'trimestral',      # Unidad de tiempo
'año_inicio': 2020,              # Para etiquetas
```

Notar que el `periodo_inicio_programa` está indexado en 0. Es decir, si
`periodo_inicio_programa = 5`, va a haber 5 períodos pre-tratamiento (0, 1, 2, 3, 4) y el
tratamiento va a comenzar en el período 5.

**Justificación:**

- **2,000 empresas**: Tamaño suficiente para obtener estimaciones precisas y permitir análisis de subgrupos. En programas reales de crédito PyME, es común tener entre 1,000 y 10,000 beneficiarios.

- **12 períodos trimestrales (3 años)**: Horizonte típico de evaluación de impacto. Permite:
  - 5 períodos pre-tratamiento para verificar tendencias paralelas
  - 7 períodos post-tratamiento para capturar efectos dinámicos

- **Inicio en período 5**: Balancea el trade-off entre:
  - Suficientes períodos pre para testear supuestos
  - Suficientes períodos post para ver efectos de mediano plazo

### 1.2 Estructura de Cohortes

```python
'n_cohortes': 4,
'cupo_por_cohorte': [80, 100, 120, 150],
```

**Justificación:**

- **4 cohortes**: Refleja la práctica común de programas que abren convocatorias periódicas (ej: una por trimestre durante el primer año).

- **Cupos crecientes [80, 100, 120, 150]**: Patrón realista donde:
  - Primera cohorte es piloto (menor cupo)
  - Cupos aumentan a medida que el programa gana tracción
  - Total: 450 tratados (~23% de las empresas)

- **Tratamiento escalonado (staggered)**: Permite usar métodos modernos de DiD (Callaway-Sant'Anna, Sun-Abraham) y estudiar heterogeneidad por timing de tratamiento.

### 1.3 Variables Observables y sus Distribuciones

Las distribuciones se eligen para replicar patrones empíricos observados en datos reales de empresas.

---

#### 1.3.1 Empleados (variable de tamaño)

```python
'empleados': {
    'distribution': 'lognormal',
    'params': {'mean': 2.3, 'sigma': 0.9},
    'min': 1, 'max': 500,
    'integer': True,
}
```

**¿Por qué Log-Normal?**

La distribución del tamaño de las empresas sigue consistentemente una **ley de potencias** o aproximadamente log-normal en datos empíricos (Gibrat, 1931; Axtell, 2001). Esto implica:

- Muchas empresas pequeñas, pocas grandes
- Cola pesada a la derecha
- Soporte en $(0, \infty)$

**Parámetros elegidos:**

Con $\mu = 2.3$ y $\sigma = 0.9$:
- Mediana = $e^{2.3} \approx 10$ empleados
- Media = $e^{2.3 + 0.9^2/2} \approx 15$ empleados
- Percentil 25 ≈ 5 empleados
- Percentil 75 ≈ 20 empleados

Esto es consistente con el universo de PyMEs (típicamente < 50 empleados).

---

#### 1.3.2 Salario Promedio

```python
'salario_promedio': {
    'distribution': 'lognormal',
    'params': {'mean': 12.0, 'sigma': 0.3},
    'min': 80000, 'max': 800000,
    'es_log': True,
}
```

**¿Por qué Log-Normal?**

Los salarios siguen una distribución log-normal por varios motivos teóricos:

1. **Proceso multiplicativo**: Los incrementos salariales suelen ser porcentuales (ej: 5% de aumento), no absolutos.
2. **Teorema del límite central multiplicativo**: El producto de muchas variables aleatorias positivas converge a log-normal.
3. **Evidencia empírica**: Bien documentado en economía laboral (Lydall, 1968; Neal & Rosen, 2000).

**Parámetros elegidos:**

Con $\mu = 12.0$ y $\sigma = 0.3$ (en logs):
- Mediana = $e^{12} \approx \$163,000$
- Rango típico: \$100,000 - \$300,000
- CV (coeficiente de variación) ≈ 30%

Valores calibrados para un país de ingreso medio (pesos argentinos o similar).

---

#### 1.3.3 Antigüedad

```python
'antiguedad': {
    'distribution': 'exponential',
    'params': {'scale': 8},
    'min': 0.5, 'max': 50,
}
```

**¿Por qué Exponencial?**

La distribución exponencial modela el tiempo transcurrido en un proceso de Poisson, lo cual es apropiado si:

1. Las empresas "nacen" a tasa constante
2. La probabilidad de observar una empresa de edad $t$ decrece exponencialmente
3. Hay mayor masa de empresas jóvenes que viejas

**Parámetros elegidos:**

Con scale = 8 años:
- Media = 8 años
- Mediana ≈ 5.5 años
- P(antigüedad > 20) ≈ 8%

Consistente con alta rotación de PyMEs y supervivencia limitada.

---

#### 1.3.4 Ratio de Formalidad

```python
'ratio_formalidad': {
    'distribution': 'beta',
    'params': {'a': 4, 'b': 2},
}
```

**¿Por qué Beta?**

La distribución Beta es la elección natural para proporciones porque:

1. **Soporte en [0, 1]**: Exactamente lo que necesitamos para un ratio
2. **Flexible**: Puede ser simétrica, sesgada a izquierda o derecha
3. **Conjugada**: Útil si se quisiera hacer inferencia bayesiana

**Parámetros elegidos:**

Con $\alpha = 4$ y $\beta = 2$:
- Media = $4/(4+2) = 0.67$ (67% formalidad promedio)
- Moda = $(4-1)/(4+2-2) = 0.75$
- Distribución sesgada a la derecha (más empresas con alta formalidad)

Refleja que PyMEs que acceden a programas de crédito tienden a ser más formales que el promedio.

---

#### 1.3.5 Variables Binarias

```python
'exportadora': {'distribution': 'bernoulli', 'params': {'p': 0.12}},
'tiene_credito': {'distribution': 'bernoulli', 'params': {'p': 0.30}},
```

**¿Por qué Bernoulli?**

Para indicadores binarios (sí/no), Bernoulli es la única opción.

**Parámetros elegidos:**

- **Exportadora (12%)**: En América Latina, típicamente 5-15% de PyMEs exportan.
- **Crédito previo (30%)**: Tasa de bancarización de PyMEs varía entre 20-40% en la región.

---

#### 1.3.6 Variables Categóricas

```python
'sector': {
    'distribution': 'categorical',
    'params': {
        'categories': ['manufactura', 'comercio', 'servicios', 'tecnologia'],
        'probs': [0.30, 0.35, 0.25, 0.10]
    },
},
'region': {
    'distribution': 'categorical',
    'params': {
        'categories': ['centro', 'norte', 'sur', 'litoral'],
        'probs': [0.45, 0.20, 0.20, 0.15]
    },
},
```

**Justificación de proporciones:**

- **Sectores**: Comercio lidera (35%), seguido de manufactura (30%), servicios (25%), tecnología (10%). Patrón típico de estructura productiva PyME.
- **Regiones**: Concentración en el centro (45%), típico de países con actividad económica concentrada en la capital/región central.

### 1.4 Variables No Observables (Fuente del Sesgo)

```python
'calidad_gerencial': {'distribution': 'normal', 'params': {'mean': 0, 'sd': 1}},
'productividad_latente': {'distribution': 'normal', 'params': {'mean': 0, 'sd': 1}},
'propension_credito': {'distribution': 'normal', 'params': {'mean': 0, 'sd': 1}},
```

**¿Por qué Normal(0,1)?**

1. **Centrada en cero**: Interpretación como desviaciones respecto a la media poblacional
2. **Varianza unitaria**: Estandarizada para facilitar interpretación de coeficientes
3. **Simétrica**: Sin supuesto a priori sobre asimetría
4. **Soporte en $\mathbb{R}$**: Permite valores extremos (muy buenos/malos gerentes)

**Interpretación:**

- **Calidad gerencial**: Capacidad de gestión, toma de decisiones, liderazgo. No observable directamente pero afecta productividad y supervivencia.

- **Productividad latente**: Factor TFP residual (Solow). Captura eficiencia no explicada por insumos observables.

- **Propensión al crédito**: Actitud hacia el endeudamiento, conocimiento financiero, necesidad de financiamiento. Principal driver de la autoselección.

**Rol en el sesgo:**

Estas variables correlacionan con:
1. La decisión de participar (vía modelo de selección)
2. Los outcomes (empleados, salarios)

Al no poder controlar por ellas, el estimador naive está sesgado.

### 1.5 Dinámica Temporal de los Outcomes

Los outcomes evolucionan según un proceso AR(1) con tendencia y ciclo:

$$Y_{it} = \rho \cdot Y_{i,t-1} \cdot (1 + \mu + \phi \cdot \xi_t + \sigma \cdot \eta_{it}) + \tau_{it}$$

```python
'dinamica_outcomes': {
    'empleados': {
        'persistencia': 0.95,      # Alta inercia
        'tendencia_base': 0.008,   # 0.8% crecimiento trimestral
        'volatilidad': 0.05,       # 5% volatilidad idiosincrática
        'efecto_ciclo': 0.015,     # Sensibilidad al ciclo
        'integer': True,
        'min': 1,
    },
    'salario_promedio': {
        'persistencia': 0.98,      # Muy alta inercia (rigidez salarial)
        'tendencia_base': 0.012,   # 1.2% (incluye inflación)
        'volatilidad': 0.025,      # Menor volatilidad
        'efecto_ciclo': 0.008,     # Menos sensible al ciclo
        'min': 50000,
    },
    'tiene_credito': {
        'persistencia': 0.90,      # Estado crediticio es sticky
        'tendencia_base': 0.01,    # Leve tendencia a bancarización
        'efecto_ciclo': 0.015,     # Crédito es procíclico
        'es_binaria': True,
    },
},
```

**Justificación de parámetros:**

| Parámetro | Empleados | Salario | Crédito | Justificación |
|-----------|-----------|---------|---------|---------------|
| Persistencia | 0.95 | 0.98 | 0.90 | Empleo es sticky; salarios más aún (rigidez nominal); crédito menos |
| Tendencia | 0.8% | 1.2% | 1.0% | Crecimiento real moderado; salarios incluyen inflación |
| Volatilidad | 5% | 2.5% | -- | Empleo más volátil que salarios (margen extensivo vs intensivo) |
| Ciclo | 1.5% | 0.8% | 1.5% | Empleo y crédito son más procíclicos que salarios |

### 1.6 Reglas de Elegibilidad

```python
'elegibilidad': [
    {'variable': 'empleados', 'operator': 'le', 'value': 100},
    {'variable': 'empleados', 'operator': 'ge', 'value': 3},
    {'variable': 'antiguedad', 'operator': 'ge', 'value': 1},
    {'variable': 'ratio_formalidad', 'operator': 'ge', 'value': 0.5},
],
```

**Justificación:**

Reglas típicas de programas PyME:

1. **3 ≤ empleados ≤ 100**: Define "pequeña y mediana empresa". Excluye microempresas (< 3) y grandes (> 100).

2. **Antigüedad ≥ 1 año**: Requisito común para demostrar track record y reducir riesgo de default.

3. **Formalidad ≥ 50%**: Asegura capacidad de cumplir obligaciones legales y tributarias.

Estas reglas generan **elegibilidad determinística**: se cumple o no, sin zona gris.

### 1.7 Modelo de Selección

La probabilidad de aplicar al programa sigue un modelo logit:

$$P(\text{Aplica}_i = 1 | \text{Elegible}_i = 1) = \Lambda(\beta_0 + \beta_X' X_i + \beta_U' U_i)$$

```python
'seleccion': {
    'intercepto_base': -1.5,
    'efectos_variables': {
        # OBSERVABLES
        'empleados': 0.008,          # Más grandes, más probabilidad
        'antiguedad': 0.015,         # Más establecidas, más probabilidad
        'exportadora': 0.35,         # Exportadoras más sofisticadas
        'tiene_credito': 0.25,       # Experiencia con crédito
        'ratio_formalidad': 0.4,     # Más formales, más acceso
        
        # NO OBSERVABLES (generan sesgo)
        'calidad_gerencial': 0.5,    # Buenos gerentes buscan oportunidades
        'productividad_latente': 0.25,  # Más productivas, más ambiciosas
        'propension_credito': 0.6,   # Principal driver de autoselección
    },
    'efectos_sector': {
        'manufactura': 0.0,          # Referencia
        'comercio': -0.15,           # Menor acceso
        'servicios': -0.1,
        'tecnologia': 0.3,           # Más propensos a buscar financiamiento
    },
    'efectos_region': {
        'centro': 0.0,               # Referencia
        'norte': -0.25,              # Menor acceso en regiones periféricas
        'sur': -0.15,
        'litoral': -0.1,
    },
},
```

**Justificación de coeficientes:**

Los signos y magnitudes reflejan patrones empíricos de acceso a crédito:

1. **Intercepto = -1.5**: Genera una tasa de aplicación base moderada (~18% con todo en cero).

2. **No observables con coeficientes positivos**: Las empresas con mejores características latentes (calidad gerencial, productividad, propensión al crédito) tienen mayor probabilidad de participar. Esto genera **sesgo de selección positivo**.

3. **Propensión al crédito (0.6)**: Es el coeficiente más alto porque captura la decisión activa de buscar financiamiento.

4. **Efectos sectoriales**: Tecnología tiene mayor acceso (+0.3) mientras comercio tiene menor (-0.15). Consistente con políticas de fomento a sectores innovadores.

5. **Efectos regionales**: Regiones periféricas tienen menor acceso, reflejando barreras de información y distancia.

### 1.8 Efectos del Tratamiento

El efecto es dinámico (crece con el tiempo) y heterogéneo (varía entre empresas):

$$\tau_i(k) = \left(\tau^{imm} + \tau^{grad} \cdot \min(k, k_{max})\right) \cdot (1 + \eta' U_i)$$

```python
'efectos_tratamiento': {
    'empleados': {
        'efecto_inmediato': 1.0,      # +1 empleado al tratarse
        'efecto_gradual': 0.5,        # +0.5 empleados por período
        'efecto_maximo': 5.0,         # Tope de +5 empleados
        'periodos_hasta_maximo': 6,   # 6 trimestres para alcanzar máximo
        'heterogeneidad': {'calidad_gerencial': 0.3},
    },
    'salario_promedio': {
        'efecto_inmediato': 0.015,    # +1.5% inmediato
        'efecto_gradual': 0.008,      # +0.8% por período
        'efecto_maximo': 0.08,        # Máximo +8%
        'periodos_hasta_maximo': 8,
        'heterogeneidad': {'calidad_gerencial': 0.01},
    },
    'tiene_credito': {
        'efecto_inmediato': 0.30,     # +30 pp inmediato
        'efecto_gradual': 0.04,       # +4 pp por período
        'efecto_maximo': 0.45,        # Máximo +45 pp
        'periodos_hasta_maximo': 4,
        'heterogeneidad': {'propension_credito': 0.08},
    },
},
```

**Justificación:**

| Outcome | Efecto inmediato | Efecto máximo (largo plazo) | Justificación |
|---------|------------------|-----------------------------|--------------|
| Empleados | +1 | +5 | Crédito permite contratar; efecto acumulativo |
| Salario | +1.5% | +8% | Productividad mejora gradualmente |
| Crédito | +30 pp | +45 pp | Acceso al programa abre puertas a más crédito |

**Heterogeneidad:**

- Empresas con mayor **calidad gerencial** aprovechan mejor el crédito → efectos mayores en empleo y salarios.
- Empresas con mayor **propensión al crédito** tienen más probabilidad de acceder a crédito adicional.

### 1.9 Ciclo Económico

```python
'ciclo_economico': {
    'activado': True,
    'volatilidad_agregada': 0.015,  # 1.5% de shock común por período
    'shocks_por_periodo': {},       # Permite especificar shocks particulares
},
```

**Justificación:**

El shock agregado $\xi_t \sim N(0, 0.015^2)$ captura fluctuaciones macroeconómicas que afectan a todas las empresas simultáneamente. Esto es importante porque:

1. Genera correlación temporal en los outcomes
2. Permite testear robustez a shocks comunes
3. Es más realista que asumir independencia temporal

La volatilidad de 1.5% por trimestre es consistente con fluctuaciones del PIB en economías emergentes.


## 2. EJECUCION DEL CODIGO

El formato del panel resultante es:
- Filas: `n_empresas` * `n_periodos`.
- Columnas: tantas como variables haya (se pueden incluir o no las no observables).

### 1. Generación de condiciones individuales
En el período `t = 0`, se genera para cada empresa el valor inicial de cada variable según
la distribución especificada en la configuración.

Esto resulta en una dataframe con `n_empresas` filas y columnas para cada variable
(observables y no observables).

### 2. Simulación de la dinámica temporal

En cada período, se genera un dataframe nuevo (en el código es `pdata`):

- Las variables fijas (que son las NO outcomes) se mantienen constantes, es decir el valor
generado en el paso 1 (Generación de condiciones individuales) se replica en cada período
para cada empresa.

- Para las outcomes:

  1. Para las empresas que en algún período anterior resultaron tratadas, se calcula el
  **efecto del tratamiento** correspondiente al período actual (que depende de cuánto
  tiempo hace que se trataron, dado por `periods_since`). Para las empresas que (aún) no
  resultaron tratadas, el efecto del tratamiento es cero. **Notar que en este paso
  aún no se aplica el efecto del tratamiento, solamente se calcula**.

  2. Se calcula el valor de la outcome en el período actual, que depende de:
     - El valor de la outcome en el período anterior (persistencia)
     - La tendencia base
     - El efecto del ciclo económico
     - Un shock idiosincrático
     - El efecto del tratamiento (si corresponde) (**acá es donde efectivamente se aplica
    el efecto del tratamiento**)

     Todos estos parámetros se indican en la configuración en `dinamica_outcomes`.

Entonces, hasta que no se llegue al primer período de tratamiento (i.e. al período
correspondiente a la primer cohorte), la dinámica temporal de las outcomes se genera
solamente a partir de la persistencia, la tendencia y el ciclo económico. A partir del
primer período de tratamiento, para cada empresa se va a generar la dinámica temporal de
las outcomes teniendo en cuenta también el efecto del tratamiento (si corresponde).

Veamos entonces qué pasa cuando se llega a un período de tratamiento:

1. Se chequea elegibilidad de cada empresa según las reglas indicadas en la configuración.
Esto se hace en el método `_check_eligibility()`, que devuelve un vector booleano
indicando si cada empresa es elegible o no. Esta elegibilidad es en base a un criterio
determinístico, es decir, se cumple o no se cumple. Cabe notar que para este
elegibilidad **se toma únicamente el valor generado en el período actual**. Además,
hasta ahora no se descartan las que ya fueron tratadas.

2. Se calcula el **efecto demostración** considerando **todas** las empresas que fueron
tratadas en algún período anterior. Esto es un número que representa cuánto afecta los
resultados vistos en los tratados anteriores sobre la probabilidad de tratarse de las
empresas elegibles en el período actual.

3. Se calcula la **probabilidad de participar** de cada empresa, teniendo en cuenta el
efecto demostración calculado anteriormente. La forma en la que se aplica este efecto
demostración es sumándoselo al "intercepto base" del modelo de selección. El intercepto
base es un término constante que representa la propensión inicial a participar antes de
sumar efectos de variables (empleados, antigüedad, sector, región, etc.). En un modelo
logístico, ese valor mueve la probabilidad "de arranque" hacia arriba o hacia abajo. La
forma en la que los valores de las distintas variables afectan la probabilidad de participar
se configura en el campo `seleccion`.

4. En este punto ya tenemos la probabilidad de participar de cada empresa. Primero, se
descartan las empresas que ya fueron tratadas en períodos anteriores: `candidates =
eligible & ~already_treated`. Luego, para cada empresa candidate, se realiza un sorteo de
una variable aleatoria uniforme entre 0 y
1. Si el valor de esta variable es menor o igual a la probabilidad de participar, entonces
esa empresa se considera participante (tratada) en el período actual. De lo contrario, se
considera no participante. En este paso se aplica el cupo del programa: si el número de
empresas que resultan participantes es mayor al cupo, entonces se ordenan según la
probabilidad de participar y se seleccionan solamente las que tienen mayor probabilidad de
participar hasta completar el cupo.
