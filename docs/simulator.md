# Informe técnico — Simulador de Panel para Programa de Crédito Empresarial

Este informe documenta el funcionamiento completo de `DataSimulator` (`simulator.py`), con las fórmulas matemáticas detrás de cada paso. Las correcciones aplicadas durante la revisión están marcadas con 🛠️, y los puntos que quedan pendientes de decisión o implementación están marcados con 🔲.

---

## 1. Estructura general del panel

El simulador genera un panel `firma × período`. Cada firma $i$ tiene:

- **Variables estáticas** $X_i$: características que no cambian en el tiempo, generadas una sola vez (`fixed_features`), salvo la excepción de `antiguedad` (ver sección 4).
- **Outcomes** $Y_{it}$: variables dinámicas que evolucionan según un proceso AR(1) (`empleados`, `salario_promedio`).
- **`inicio_firma`**: período en el que la firma empieza a existir dentro del panel. Se sortea de forma independiente a cualquier característica económica de la firma:

$$
\text{inicio\_firma}_i \sim \text{Uniforme}\{0, 1, \ldots, t_0 - \text{min\_periodos\_pre\_programa}\}
$$

donde $t_0$ es `periodo_inicio_programa`. Esto garantiza que toda firma tenga al menos `min_periodos_pre_programa` períodos de historia antes de que el programa arranque.

El panel final se recorta para que cada firma solo tenga filas desde su `inicio_firma` en adelante — antes de eso, la firma "no existe" en el panel observado, aunque sus variables ya hayan sido generadas.

---

## 2. Generación de condiciones iniciales

Cada variable estática y cada outcome se generan en $t=0$ según su distribución especificada en `config['variables']`. Las distribuciones disponibles son:

| Distribución | Parámetros | Media $\mathbb{E}[X]$ | Uso en el config |
|---|---|---|---|
| Exponencial | `scale` | $\text{scale}$ | `antiguedad` (scale=8) |
| Categórica | `categories`, `probs` | — (variable discreta) | `sector`, `region` |
| Bernoulli | `p` | $p$ | `exportadora` (p=0.12) |
| Beta | `a`, `b` | $\frac{a}{a+b}$ | `ratio_formalidad` (a=4,b=2 → 0.67) |
| Normal | `mean`, `sd` | $\text{mean}$ | variables no observables (mean=0) |
| Lognormal | `mean`, `sigma` | $e^{\text{mean}+\sigma^2/2}$ | `empleados`, `salario_promedio` (solo para $t=0$, antes del burn-in) |

Después de generar los valores, se aplican en orden: exponenciación si `es_log`, clip a `[min, max]`, y redondeo a entero si `integer`.

---

## 3. Shock macroeconómico $\lambda_t$ — proceso AR(1) puro

El ciclo económico se genera **una sola vez** para toda la simulación, independiente de las firmas:

$$
\lambda_t = \rho_\lambda \cdot \lambda_{t-1} + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, \sigma_\lambda^2)
$$

con $\rho_\lambda = $ `ciclo_economico.persistencia` y $\sigma_\lambda = $ `ciclo_economico.volatilidad`.

**Es un AR(1) sin intercepto**, por lo tanto:

$$
\mathbb{E}[\lambda_t] = 0 \qquad \text{(para todo } t\text{)}
$$

La condición inicial se sortea directamente desde la distribución estacionaria del proceso:

$$
\lambda_0 \sim \mathcal{N}\left(0, \ \frac{\sigma_\lambda^2}{1-\rho_\lambda^2}\right)
$$

y además se generan `N_BURN_IN = 100` períodos adicionales que se descartan, para asegurar que la serie usada esté bien mezclada.

> 🔲 **Punto evaluado y descartado como corrección.** Durante el burn-in de los *outcomes* (sección 5), `_evolve_outcome` se llama siempre con `t=0`, por lo que usa `aggregate_shocks[0]` como una constante repetida 100 veces en lugar de un shock variable. Se evaluó como corrección, pero se descartó: `shocks[0]` ya es un valor legítimo sorteado desde la distribución estacionaria (no es un valor "sin calentar"), y la magnitud del efecto es negligible ($\sigma_\lambda \approx 0.02$ en la calibración actual) frente a la escala de los outcomes. **No se modificó código por este punto.**

---

## 4. Antigüedad dinámica 🛠️ CORREGIDO

**Problema original:** `antiguedad` estaba declarada como `fixed_feature`, por lo que nunca se actualizaba — una firma conservaba el mismo valor de antigüedad en el período 0 y en el período 20, como si el tiempo no pasara. Esto rompía la regla de elegibilidad `antiguedad ≥ 1` para firmas jóvenes de forma permanente.

**Corrección aplicada:** se ancla el envejecimiento a `inicio_firma`, no al índice absoluto `t`, para no generar inconsistencia con firmas que entran tarde al panel:

$$
\text{antiguedad}_i(t) = \text{antiguedad}_{i,0} + (t - \text{inicio\_firma}_i)
$$

Interpretación: $\text{antiguedad}_{i,0}$ es la edad real de la firma en el momento en que empieza a ser observada en el panel (variable con sentido económico propio, independiente de `inicio_firma`); a partir de ahí envejece un período genuino por cada período transcurrido dentro del panel.

Esto asume implícitamente que **1 período del panel = 1 año** (consistente con que `antiguedad` esté expresada en años, `scale=8`, rango `[0.5, 50]`).

> 🔲 **Pendiente menor:** esta unidad temporal (1 período = 1 año) queda implícita en el código y no está documentada explícitamente en `data_config.py`. Se sugirió agregar `'periodo_en_anios': 1` como comentario/parámetro explícito, pero no es bloqueante.

El valor se guarda como columna `antiguedad_{t}` en `firms`, por lo que las tres funciones que ya usan el patrón `col = f'{var}_{t}' if ... else f'{var}_0'` (`_check_eligibility`, `_evolve_outcome`, `_compute_propensity`) levantan automáticamente el valor envejecido.

---

## 5. Evolución de los outcomes — proceso AR(1) con efectos fijos y temporales

Este es el corazón del DGP, implementado en `_evolve_outcome`. Para cada outcome $Y \in \{\text{empleados}, \text{salario\_promedio}\}$:

$$
Y_{it} = \rho \cdot Y_{i,t-1} \;+\; \alpha_i \;+\; \lambda_t \;+\; \varepsilon_{it} \;+\; \tau_{it}
$$

donde:

### 5.1 — Efectos fijos $\alpha_i$

$$
\alpha_i = \sum_{k \,\in\, \text{numéricas}} \beta_k \cdot X_{ik} \;+\; \sum_{c \,\in\, \text{categóricas}} \beta_{c,\,\text{categoría}(i)}
$$

Se lee de `dinamica_outcomes[outcome].efectos_variables`. Para variables numéricas es un producto directo; para variables categóricas (`sector`, `region`) se usa el efecto correspondiente a la categoría de la firma. **Importante:** el valor de cada variable se toma en $t-1$ si existe esa columna, si no cae al valor fijo en $t=0$ — en la práctica, para las variables estáticas esto siempre resuelve al valor de $t=0$ (o al envejecido, en el caso de antigüedad).

Como derivamos analíticamente, la media de largo plazo de la firma es:

$$
\mu_i = \frac{\alpha_i}{1-\rho}
$$

> 🔲 **Pendiente de implementar:** se decidió que `sector` y `region` no deberían afectar la media de largo plazo (los valores actuales del config eran arbitrarios). Para lograrlo, los efectos de cada categoría deben **centrarse** para que su contribución esperada sea cero:
>
> $$
> \sum_s \beta_s \cdot P(\text{categoría}=s) = 0, \qquad \beta_s^{\text{centrado}} = \beta_s^{\text{original}} - \sum_{s'} \beta_{s'}^{\text{original}} \cdot P(s')
> $$
>
> Los valores centrados ya fueron calculados para sector y región en `empleados` y `salario_promedio`, pero **no están todavía aplicados en `data_config.py`**.

### 5.2 — Efectos temporales $\lambda_t$

Shock macro común a todas las firmas (sección 3), activo solo si `ciclo_economico.activado = True`.

### 5.3 — Error idiosincrático $\varepsilon_{it}$

$$
\varepsilon_{it} \sim \mathcal{N}(0, \ \text{volatilidad}^2), \qquad \text{volatilidad} = \text{dinamica\_outcomes[outcome].volatilidad}
$$

Independiente entre firmas y entre períodos.

### 5.4 — Media y varianza estacionarias del proceso completo

Como $\mathbb{E}[\lambda_t] = 0$ y $\mathbb{E}[\varepsilon_{it}]=0$:

$$
\mathbb{E}[Y_{it}] \xrightarrow{t\to\infty} \mu_i = \frac{\alpha_i}{1-\rho}
$$

La condición de estacionariedad del proceso es $|\rho| < 1$. La velocidad de convergencia depende de $\rho^t \to 0$; la vida media (períodos para recorrer la mitad de la distancia al valor estacionario) es:

$$
\text{vida media} = \frac{\ln(0.5)}{\ln(\rho)}
$$

### 5.5 — Efecto del tratamiento $\tau_{it}$

Se aplica de dos formas posibles, según `dinamica_outcomes[outcome].efecto_tratamiento`:

$$
Y_{it} = \begin{cases}
\text{base}_{it} + \tau_{it} & \text{si `aditivo'} \quad (\text{ej. empleados}) \\[4pt]
\text{base}_{it} \times (1+\tau_{it}) & \text{si `porcentual'} \quad (\text{ej. salario\_promedio})
\end{cases}
$$

El cálculo de $\tau_{it}$ se detalla en la sección 6.

---

## 6. Efecto dinámico del tratamiento

Implementado en `_compute_treatment_effect`. Para una firma tratada en el período $\text{periodo\_tratamiento}_i$, con $k$ = períodos transcurridos desde el tratamiento:

$$
k = t - \text{periodo\_tratamiento}_i, \qquad k^* = \min(k, \ k_{\max})
$$

$$
\tau_{\text{base}} = \min\big(\tau_{\text{imm}} + \tau_{\text{grad}} \cdot k^*, \ \ \tau_{\text{max}}\big)
$$

Con heterogeneidad individual según variables no observables (por ejemplo `calidad_gerencial`):

$$
h_i = \text{clip}\left(\sum_v \gamma_v \cdot \text{clip}(X_{iv}, -2, 2), \ -0.4,\ 0.4\right)
$$

$$
\tau_i(k) = \tau_{\text{base}} \cdot (1 + h_i)
$$

Firmas no tratadas en el período $t$ reciben $\tau_i = 0$.

---

## 7. Reglas de elegibilidad

Implementado en `_check_eligibility`. Todas las condiciones deben cumplirse simultáneamente (AND):

$$
\text{Elegible}_i(t) = \bigwedge_{r \,\in\, \text{elegibilidad}} \Big( X_{i,\text{var}(r)}(t) \ \ \text{op}(r) \ \ \text{valor}(r) \Big)
$$

con operadores `ge` ($\geq$), `le` ($\leq$), `gt` ($>$), `lt` ($<$). Usa el valor de la variable **en el período actual** $t$ si existe esa columna (por eso `antiguedad` dinámica importa acá), si no cae al valor fijo de $t=0$.

Config actual:

$$
\text{empleados} \in [3, 100], \qquad \text{antiguedad} \geq 1, \qquad \text{ratio\_formalidad} \geq 0.5
$$

Como `empleados` es un outcome que evoluciona vía AR(1), la elegibilidad genera un **loop de retroalimentación**: $Y(t-1) \to \text{elegibilidad}(t) \to \text{tratamiento}(t) \to Y(t)$.

---

## 8. Efecto demostración 🛠️ CORREGIDO (escala)

Implementado en `_compute_demonstration_effect`. Mide cuánto el desempeño de firmas ya tratadas influye en la probabilidad de que firmas nuevas entren al programa.

Para cada firma ya tratada (`tratado=True` y `periodo_tratamiento < t`), se pondera su crecimiento por un factor de decaimiento según cuánto tiempo pasó desde que entró:

$$
w_i = \delta^{\,(t - \text{periodo\_tratamiento}_i)}, \qquad \delta = \text{efecto\_demostracion.decaimiento}
$$

**Problema original:** el crecimiento se calculaba siempre como fracción relativa, $(Y_t - Y_0)/(|Y_0|+1)$, y se dividía por `efecto_maximo` sin distinguir que ese `efecto_maximo` está en **unidades absolutas** para outcomes aditivos (ej. 5 empleados) y en **fracción** para outcomes porcentuales (ej. 0.08 = 8%). Esto hacía que `empleados` pesara ~60 veces menos de lo esperado en el promedio final.

**Corrección aplicada** — el cálculo de crecimiento ahora depende del tipo de efecto de tratamiento del outcome, para quedar en la misma unidad que su `efecto_maximo`:

$$
\text{growth}_{i,\text{outcome}} =
\begin{cases}
Y_{it} - Y_{i,\text{periodo\_tratamiento}} & \text{si `aditivo'} \\[6pt]
\dfrac{Y_{it} - Y_{i,\text{periodo\_tratamiento}}}{|Y_{i,\text{periodo\_tratamiento}}| + \epsilon} & \text{si `porcentual'}
\end{cases}
$$

$$
\text{growth}_{i,\text{outcome}}^{\text{norm}} = \frac{\text{growth}_{i,\text{outcome}}}{\text{efecto\_maximo}_{\text{outcome}}}
$$

De esta forma, una firma que alcanza exactamente el efecto máximo del programa da $\text{growth}^{\text{norm}} = 1.0$, sea el outcome `empleados` o `salario_promedio`.

Luego se promedia (ponderado por $w_i$) dentro de cada outcome, y se promedia simple entre outcomes:

$$
\overline{\text{growth}}_{\text{outcome}} = \frac{\sum_i w_i \cdot \text{growth}_{i,\text{outcome}}^{\text{norm}}}{\sum_i w_i}, \qquad
\overline{\text{growth}} = \frac{1}{|\text{outcomes}|}\sum_{\text{outcome}} \overline{\text{growth}}_{\text{outcome}}
$$

Efecto final, con sensibilidad y ruido:

$$
\text{demo}_t = \text{clip}\Big(\underbrace{s \cdot \overline{\text{growth}}}_{\text{efecto base}} + \underbrace{\mathcal{N}(0,\ \sigma_{\text{ruido}})}_{\text{ruido}}, \ -0.5, \ 1.0\Big)
$$

con $s = $ `sensibilidad`. Este `demo_t` es un escalar (no varía por firma) que se suma al intercepto del propensity score en el período $t$.

---

## 9. Propensity score

Implementado en `_compute_propensity`. Es un modelo logit:

$$
z_i(t) = \beta_0 + \text{demo}_t + \sum_k \gamma_k \cdot X_{ik}(t) + \gamma_{\text{sector}(i)} + \gamma_{\text{región}(i)} + \text{trend}_i(t)
$$

$$
p_i(t) = \sigma(z_i(t)) = \frac{1}{1+e^{-z_i(t)}}
$$

con $z$ recortado a $[-10, 10]$ antes de la sigmoide, para evitar overflow numérico. $\beta_0 = $ `intercepto_base`, $\gamma_k = $ `efectos_variables` (incluye tanto observables como no observables — estas últimas son la fuente de sesgo de selección no controlable). Cada $X_{ik}(t)$ se toma en el período actual si existe esa columna dinámica, si no cae al valor fijo de $t=0$.

### 9.1 — Componente de historia: tendencia (implementación actual)

$$
\text{trend}_i(t) = \sum_{\text{outcome}} c_{\text{outcome}} \cdot \frac{Y_{i,t} - Y_{i,t-\text{ventana}}}{|Y_{i,t-\text{ventana}}| + 1}
$$

Esta es la función `_compute_outcome_trend` — mide la **pendiente** de la trayectoria entre dos puntos separados por `ventana` períodos.

> 🔲 **Pendiente — decisión tomada, no implementada.** Se decidió reemplazar este mecanismo de tendencia por uno de **nivel + estabilidad**, que resume toda la ventana pre-tratamiento (no solo dos puntos) en dos estadísticos por firma:
>
> $$
> \overline{Y}_i = \frac{1}{k}\sum_{j=1}^{k} Y_{i,t-j}, \qquad s_i^2 = \frac{1}{k-1}\sum_{j=1}^{k}(Y_{i,t-j}-\overline{Y}_i)^2
> $$
>
> normalizados cross-seccionalmente (z-score entre firmas):
>
> $$
> \tilde{Y}_i = \frac{\overline{Y}_i - \text{media}_i(\overline{Y})}{\text{sd}_i(\overline{Y})}, \qquad \tilde{s}_i^2 = \frac{s_i^2 - \text{media}_i(s^2)}{\text{sd}_i(s^2)}
> $$
>
> $$
> \text{efecto}_i(t) = \gamma_{\text{nivel}} \cdot \tilde{Y}_i \; + \; \gamma_{\text{estabilidad}} \cdot (-\tilde{s}_i^2)
> $$
>
> La función `_compute_outcome_level_stability` que implementa esto fue redactada y entregada, pero **no fue aplicada todavía en `simulator.py`** (`_compute_outcome_trend` sigue vigente sin cambios). Tampoco se agregó la clave `historial_outcomes` con `coef_nivel` / `coef_estabilidad` en `data_config.py` — actualmente, como esa clave no existe en el config, ni la función de tendencia ni la de nivel+estabilidad aportan nada al propensity score.

---

## 10. Asignación de tratamiento con cupo

Implementado en `_assign_treatment`, ejecutado una vez por cohorte (cuando $t - t_0 \in [0, n\_cohortes)$):

**Paso 1 — Candidatos:** elegibles y no tratados previamente.

**Paso 2 — Aplicación (Bernoulli):**

$$
\text{aplica}_i \sim \text{Bernoulli}(p_i(t))
$$

Se simula generando $u_i \sim \text{Uniforme}(0,1)$ y comparando $u_i < p_i(t)$.

**Paso 3 — Split 50/50 aleatorio** entre aplicantes: la mitad son candidatos a tratamiento, la otra mitad quedan como pool de control.

**Paso 4 — Recorte por cupo:** si los candidatos a tratamiento superan el cupo de la cohorte, se ordenan por un score con ruido y se toman los primeros:

$$
\text{score}_i = p_i(t) + \mathcal{N}(0, 0.1)
$$

$$
\text{tratados} = \text{top-}\text{cupo}\big(\text{score}_i\big)
$$

El ruido introduce aleatoriedad adicional en el margen, evitando que la selección sea un corte determinístico puro sobre el propensity score.

---

## 11. Construcción del panel final

1. Se generan condiciones iniciales y se corre el burn-in (100 períodos descartados) para que los outcomes converjan a su distribución estacionaria antes de $t=0$ observable.
2. Para cada $t = 0, \ldots, n\_periodos-1$: se actualiza antigüedad, se evolucionan los outcomes (observado y contrafactual), se asigna tratamiento si corresponde a una cohorte.
3. El contrafactual $Y_{it}^{\text{cf}}$ sigue la misma trayectoria que $Y_{it}$ hasta el momento en que la firma es tratada — a partir de ahí, $Y^{\text{cf}}$ nunca recibe $\tau_{it}$, mientras que $Y^{\text{obs}}$ sí. Esta es la referencia (*ground truth*) contra la que se puede evaluar cualquier método de estimación del efecto del tratamiento.
4. El panel se recorta a $t \geq \text{inicio\_firma}$ por firma.
5. Firmas que fueron control en una cohorte y luego tratadas en una posterior pierden su condición de control retroactivamente en todas sus filas (para no contaminar el grupo de control con firmas que terminan tratadas).

---

## 12. Resumen de correcciones

| # | Ítem | Estado | Efecto |
|---|---|---|---|
| 1 | Clip de mínimo / redondeo a entero no se aplicaba (código muerto en un loop) | 🛠️ Corregido | `empleados` ahora queda entero; contrafactual correctamente clippeado |
| 2 | `antiguedad` fija en el tiempo, sin envejecer | 🛠️ Corregido | Envejece como `antiguedad_0 + (t - inicio_firma)` |
| 3 | `tiene_credito` referenciado en config pero nunca declarado como variable/outcome | 🛠️ Corregido (config) | Removido de `seleccion.efectos_variables` y `efectos_tratamiento` |
| 4 | Escala inconsistente en efecto demostración (fracción ÷ unidad absoluta) | 🛠️ Corregido | `empleados` y `salario_promedio` ahora comparables en el efecto demostración |
| 5 | Shock macro constante durante burn-in de outcomes | 🔲 Evaluado, descartado | Efecto negligible; no se modificó código |
| 6 | Sector/región afectan la media de largo plazo (valores arbitrarios) | 🔲 Pendiente | Valores centrados ya calculados, no aplicados en `data_config.py` |
| 7 | Mecanismo temporal del propensity: tendencia vs. nivel+estabilidad | 🔲 Pendiente | Función `_compute_outcome_level_stability` redactada, no aplicada; falta también `historial_outcomes` en config |
| 8 | Documentar unidad temporal implícita (1 período = 1 año) | 🔲 Pendiente menor | No bloqueante |

---

## 13. Lo que falta antes de calibrar con el HTML

Para que la calibración con `calibracion_interactiva.html` sea coherente con el simulador real:

1. Aplicar los efectos centrados de sector/región en `data_config.py` (punto 6).
2. Decidir si se aplica nivel+estabilidad ahora o se pospone, y en ese caso completar `historial_outcomes` en el config (punto 7).
3. Fijar $\rho$, $\mu_{\text{target}}$, $\sigma_{\text{target}}$ por outcome en el HTML y exportar el config resultante.
4. Correr `main.py` y verificar empíricamente: fracción de elegibles, fracción tratada por cohorte, medias observadas vs. targets.