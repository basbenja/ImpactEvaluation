# Dinámica del Modelo de Simulación

## Índice

1. [Estructura general del modelo](#1-estructura-general-del-modelo)
2. [Variables del modelo](#2-variables-del-modelo)
3. [Evolución de los outcomes](#3-evolución-de-los-outcomes)
4. [Asignación de tratamiento](#4-asignación-de-tratamiento)
5. [Efectos del tratamiento](#5-efectos-del-tratamiento)
6. [Análisis de escala y calibración](#6-análisis-de-escala-y-calibración)

---

## 1. Estructura general del modelo

El simulador genera un panel sintético de firmas que participan (o no) en un programa de crédito. El panel es de formato *long*: una fila por cada par `(firma, período)`.

Cada firma existe desde un período `inicio_firma` y es observada hasta el último período `T`. El programa comienza en `t₀` y asigna tratamiento en cohortes sucesivas: una cohorte por período, desde `t₀` hasta `t₀ + n_cohortes - 1`.

Las firmas se clasifican en tres tipos:

| Tipo | Definición |
|---|---|
| **Tratados (T)** | Ingresaron al programa en alguna cohorte |
| **Controles (C)** | Cumplían condiciones para ingresar pero no lo hicieron |
| **NiNis** | Nunca elegibles o nunca aplicaron |

---

## 2. Variables del modelo

### 2.1 Características fijas

Se generan una vez en `t = 0` y no cambian. Se dividen en:

**Observables** — visibles en los datos de panel:

| Variable | Distribución | Descripción |
|---|---|---|
| `antiguedad` | Exponencial(escala=8) | Años de existencia de la firma |
| `sector` | Categórica | Manufactura / Comercio / Servicios / Tecnología |
| `region` | Categórica | Centro / Norte / Sur / Litoral |
| `exportadora` | Bernoulli(p=0.12) | Si la firma exporta |
| `ratio_formalidad` | Beta(4, 2) | Proporción de empleo formal |

**No observables** — existen en la simulación pero no aparecen en los datos exportados:

| Variable | Distribución | Descripción |
|---|---|---|
| `calidad_gerencial` | Normal(0, 1) | Capacidad de gestión de la dirección |
| `productividad_latente` | Normal(0, 1) | Productividad no medida directamente |
| `propension_credito` | Normal(0, 1) | Predisposición al uso del crédito |

Las variables no observables son clave para generar **sesgo de selección**: influyen tanto en los outcomes como en la probabilidad de participar en el programa, pero el evaluador no las puede controlar.

### 2.2 Outcomes

Variables que evolucionan en el tiempo y sobre las que el programa busca tener efecto:

| Variable | Distribución inicial | Rango | Descripción |
|---|---|---|---|
| `empleados` | Lognormal(2.3, 0.9) | 1–500, entero | Número de empleados |
| `salario_promedio` | Lognormal(12.0, 0.3) | 80k–800k | Salario promedio mensual |

---

## 3. Evolución de los outcomes

### 3.1 Modelo AR(1)

Cada outcome evoluciona según un proceso autorregresivo de orden 1:

$$Y_{i,t} = \rho \cdot Y_{i,t-1} + c_i + \varepsilon_{i,t} + \tau_{i,t}$$

donde:

| Término | Nombre | Descripción |
|---|---|---|
| $\rho$ | Persistencia | Cuánto del valor anterior se mantiene (0 < ρ < 1) |
| $c_i$ | Efecto fijo de firma | Contribución de las características fijas de la firma |
| $\varepsilon_{i,t}$ | Ruido idiosincrático | Shock aleatorio por firma y período: $\varepsilon \sim \mathcal{N}(0, \sigma)$ |
| $\tau_{i,t}$ | Efecto del tratamiento | Cero si la firma no fue tratada (ver Sección 5) |

### 3.2 Efecto fijo de firma

El término $c_i$ captura la heterogeneidad permanente entre firmas:

$$c_i = \sum_k \beta_k \cdot X_{ik}$$

Los $X_{ik}$ son las características fijas de la firma (observables y no observables). Como son constantes en el tiempo pero distintas entre firmas, $c_i$ actúa como un **efecto fijo individual**: desplaza permanentemente la trayectoria de cada firma respecto al promedio.

- Variables con $E[X] \neq 0$ (como `antiguedad`, `exportadora`, `ratio_formalidad`) contribuyen al **nivel medio** del outcome.
- Variables con $E[X] = 0$ (como `calidad_gerencial`, `productividad_latente`) no afectan la media, pero crean **dispersión**: firmas con alta calidad gerencial tienen trayectorias sistemáticamente por encima del promedio.

### 3.3 Estado estacionario

Con $\rho < 1$, el AR(1) converge a un estado estacionario específico para cada firma:

$$Y^*_i = \frac{c_i}{1 - \rho}$$

El factor $\frac{1}{1-\rho}$ amplifica el efecto de $c_i$. Con $\rho = 0.95$, ese multiplicador es 20. Con $\rho = 0.98$, es 50.

La velocidad de convergencia al estado estacionario está dada por la **vida media**:

$$t_{1/2} = \frac{\log(0.5)}{\log(\rho)}$$

| $\rho$ | Vida media | % convergido en t=40 |
|---|---|---|
| 0.90 | 7 períodos | ~99% |
| 0.95 | 14 períodos | ~87% |
| 0.98 | 34 períodos | ~55% |
| 0.99 | 69 períodos | ~33% |

Si los valores iniciales $Y_{i,0}$ están lejos de $Y^*_i$, el panel exhibirá una **deriva sistemática** (tendencia al alza o a la baja) que no refleja ningún fenómeno económico real, sino simplemente el ajuste al estado estacionario.

### 3.4 Varianza en estado estacionario

La varianza cross-seccional de los outcomes en estado estacionario se descompone en dos fuentes:

$$\text{Var}(Y^*_i) = \underbrace{\frac{\text{Var}(c_i)}{(1-\rho)^2}}_{\text{heterogeneidad permanente}} + \underbrace{\frac{\sigma^2_\varepsilon}{1 - \rho^2}}_{\text{ruido idiosincrático}}$$

- **Heterogeneidad permanente**: dispersión explicada por diferencias fijas entre firmas (covariables).
- **Ruido idiosincrático**: dispersión generada por shocks aleatorios período a período.

La `volatilidad` en el config corresponde a $\sigma_\varepsilon$.

---

## 4. Asignación de tratamiento

### 4.1 Elegibilidad (condiciones determinísticas)

En cada período de cohorte, primero se evalúan condiciones determinísticas. Una firma es **elegible** si y solo si cumple **todas** las reglas definidas en `elegibilidad`. En el config actual:

- `empleados ≥ 3`
- `empleados ≤ 100`
- `antiguedad ≥ 1`
- `ratio_formalidad ≥ 0.5`

Estas reglas usan los valores del período actual, no los iniciales.

### 4.2 Propensity score (probabilidad de aplicar)

Cada firma elegible decide si aplica al programa con probabilidad:

$$p_i = \sigma\left(z_i\right), \quad z_i = \alpha_0 + \sum_k \gamma_k \cdot X_{ik} + \text{efecto\_sector} + \text{efecto\_región} + \delta_t$$

donde $\sigma(\cdot)$ es la función logística y $\delta_t$ es el efecto demostración del período (ver §4.4).

**Crucialmente**, $z_i$ incluye las variables **no observables** (`calidad_gerencial`, `productividad_latente`, `propension_credito`). Esto genera selección endógena: las firmas con mejor gestión son más propensas a participar *y* tienen mejores outcomes de base.

### 4.3 Asignación con cupo

El proceso de asignación es:

1. Firmas elegibles y no tratadas previamente → candidatas
2. Cada candidata aplica con probabilidad $p_i$ (Bernoulli)
3. Los aplicantes se dividen 50/50 aleatoriamente en tratados y controles
4. Si hay más tratados que cupo disponible, se selecciona por score (propensity + ruido)

Los controles son firmas que aplicaron pero no recibieron tratamiento en esa cohorte.

### 4.4 Efecto demostración

El término $\delta_t$ captura un efecto de contagio entre cohortes: si las firmas ya tratadas muestran crecimiento visible, las firmas aún no tratadas incrementan su propensión a participar en cohortes siguientes.

Se calcula como el crecimiento promedio ponderado (con decaimiento temporal) de los outcomes de firmas ya tratadas, normalizado por el efecto máximo esperado del tratamiento.

---

## 5. Efectos del tratamiento

El efecto del tratamiento sobre la firma $i$ en el período $t$ (siendo $k = t - t_{\text{tratamiento}}$ los períodos desde el tratamiento) es:

$$\tau_i(k) = \left[\tau_{\text{imm}} + \tau_{\text{grad}} \cdot \min(k, k_{\max})\right] \cdot (1 + h_i)$$

donde:

| Parámetro | Descripción |
|---|---|
| $\tau_{\text{imm}}$ | Efecto inmediato al ingresar al programa |
| $\tau_{\text{grad}}$ | Incremento gradual por período adicional |
| $k_{\max}$ | Períodos hasta alcanzar el efecto máximo |
| $h_i$ | Heterogeneidad individual (función de covariables) |

El efecto se aplica de forma **aditiva** o **porcentual** según el outcome:

- `empleados`: aditivo → `Y_t = Y_t_base + τ_i(k)`
- `salario_promedio`: porcentual → `Y_t = Y_t_base × (1 + τ_i(k))`

La heterogeneidad $h_i$ se construye como combinación lineal de covariables (por defecto, `calidad_gerencial`), haciendo que el efecto sea mayor en firmas con mejor capacidad de gestión.

---

## 6. Análisis de escala y calibración

### 6.1 El problema de escala

El efecto de una covariable sobre el outcome **no depende solo del coeficiente $\beta_k$**, sino del producto:

$$\text{efecto LR} = \frac{\beta_k \cdot E[X_k]}{1 - \rho}$$

Esto tiene dos consecuencias importantes:

1. El mismo coeficiente produce efectos muy distintos según la escala de la variable.
2. El nivel medio del outcome en estado estacionario es $E[Y^*] = \frac{\sum_k \beta_k \cdot E[X_k]}{1 - \rho}$.

Si $E[Y^*]$ difiere mucho de $E[Y_0]$, el panel tendrá deriva. Si los efectos LR son minúsculos en comparación con la escala del outcome, los coeficientes son irrelevantes y cambiarlos no altera los resultados observables.

**Ejemplo con los valores actuales del config:**

| Variable | $E[X]$ | $\beta_{emp}$ | Efecto LR (empleados) | $\beta_{sal}$ | Efecto LR (salario) |
|---|---|---|---|---|---|
| `antiguedad` | 8.0 | 0.80 | **+128** | 0.06 | +24 |
| `exportadora` | 0.12 | 0.80 | +1.9 | 0.80 | +4.8 |
| `ratio_formalidad` | 0.67 | 0.50 | +6.7 | 0.50 | +16.7 |
| `calidad_gerencial` | 0 | 0.30 | 0 (solo dispersión) | 0.30 | 0 |

- `salario_promedio` tiene estado estacionario ≈ **45 pesos**, mientras que el valor inicial es ≈ **170.000**. El panel muestra una caída libre de 55% en 40 períodos sin ningún tratamiento.
- `empleados` tiene estado estacionario ≈ 137, mientras el valor inicial es ≈ 15. El panel muestra un crecimiento de 8x sin tratamiento.
- La `volatilidad = 0.025` es negligible para ambos outcomes: para `empleados` (entero), la probabilidad de que el ruido cambie el valor en 1 unidad es ≈ $10^{-7}$.

### 6.2 Metodología de calibración

Para calibrar correctamente los coeficientes, el proceso recomendado es:

#### Paso 1 — Definir distribución objetivo en estado estacionario

Anclar en criterios de elegibilidad y/o datos reales del programa:

| Outcome | Target media | Target std | Justificación |
|---|---|---|---|
| `empleados` | 30–50 | 20–30 | Centro del rango elegible (3–100) |
| `salario_promedio` | según contexto | según contexto | Estadísticas SME del país/sector |

#### Paso 2 — Elegir $\rho$ por criterio económico

¿Qué tan persistente es el empleo/salario entre períodos? Considerar:
- $\rho = 0.95$: moderadamente persistente, vida media ≈ 14 períodos
- $\rho = 0.98$: muy persistente, vida media ≈ 34 períodos

Con pocos períodos de observación (e.g., $T = 12$), ρ alto implica que casi no hay convergencia y los valores iniciales dominan.

#### Paso 3 — Calcular el presupuesto de coeficientes

Para que $E[Y^*] = \mu_{\text{target}}$, la suma de efectos debe satisfacer:

$$\sum_k \beta_k \cdot E[X_k] = \mu_{\text{target}} \cdot (1 - \rho)$$

Este es el **presupuesto total de $c$** a repartir entre variables.

#### Paso 4 — Asignar presupuesto según importancia relativa

Decidir qué proporción del presupuesto explica cada variable, basado en criterio económico:

$$\beta_k = \frac{s_k \cdot \text{presupuesto}}{E[X_k]}$$

donde $s_k$ es la proporción asignada a la variable $k$ ($\sum_k s_k = 1$).

#### Paso 5 — Calibrar volatilidad desde el std objetivo

La varianza del outcome en estado estacionario debe igualar $\sigma^2_{\text{target}}$. Si se asigna una fracción $f$ al ruido idiosincrático:

$$\sigma_\varepsilon = \sqrt{f \cdot \sigma^2_{\text{target}} \cdot (1 - \rho^2)}$$

Para `empleados` con $\sigma_{\text{target}} = 25$ y $\rho = 0.95$:

$$\sigma_\varepsilon = \sqrt{0.3 \cdot 625 \cdot 0.0975} \approx 4.3$$

En lugar de `volatilidad = 0.025`, el valor correcto es del orden de **3–7**.

#### Paso 6 — Calibrar variables con $E[X] = 0$ desde el std objetivo

Las variables normales no afectan la media pero sí la dispersión. Su contribución a $\text{Var}(c_i)$ es $\beta_k^2 \cdot \text{Var}(X_k) = \beta_k^2$. Para asignarles una fracción $g$ del std objetivo:

$$\beta_k = \sqrt{g_k \cdot \sigma^2_{\text{target}} \cdot (1-\rho)^2}$$

Esto define cuánta varianza de los outcomes es explicada por los no observables — parámetro crítico para la dificultad del problema de evaluación de impacto.

#### Paso 7 — Verificar por simulación e iterar

Ejecutar el notebook de sensibilidad y verificar:

- [ ] Trayectoria media plana (sin deriva) → condiciones iniciales cercanas al estado estacionario
- [ ] Std cross-seccional coincide con el target
- [ ] Trayectorias individuales muestran variación realista (volatilidad no nula)
- [ ] Fracción de firmas elegibles razonable (30–60% de la población)
- [ ] Efecto del tratamiento detectable: $\tau_{\max} / \sigma_{\text{target}} > 0.1$

### 6.3 Condición de consistencia interna

El sistema es internamente consistente cuando:

$$E[Y_{i,0}] \approx E[Y^*_i] = \frac{\sum_k \beta_k \cdot E[X_k]}{1-\rho}$$

Si esta condición no se cumple, la simulación introduce tendencias artificiales que contaminan la señal del tratamiento y hacen más difícil (o más fácil) la identificación del efecto causal, sin que eso refleje nada del diseño del programa.
