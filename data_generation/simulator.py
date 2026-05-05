import json
import os
import numpy as np
import pandas as pd
import sys

from datetime import datetime
from pandera.typing import DataFrame as PanderaDataFrame
from scipy.special import expit     # logit
from typing import Dict

sys.path.append('..')
from config import DATA_DIR

from data_generation.panel_schema import PanelSchema

class DataSimulator:
    """
    Simulador de panel para programa de crédito empresarial.
    """
    def __init__(self, config: Dict):
        self.config = config
        # default_rng (random number generator) para reproducibilidad
        self.rng = np.random.default_rng(config.get('random_seed'))

        cupos = config['cupo_por_cohorte']
        self.cupos = cupos if isinstance(cupos, list) else [cupos] * config['n_cohortes']

        self.features = config['variables']

        # Generar shocks agregados
        vol = config['ciclo_economico']['volatilidad_agregada']
        # Distribución normal con media 0 y desviación vol
        # Un shock por cada período
        self.aggregate_shocks = self.rng.normal(0, vol, config['n_periodos'])

        self.treatment_history = {}

        # Outcomes: son las variables sobre las que se espera que el programa
        # tenga un efecto
        self.outcomes = list(config['dinamica_outcomes'].keys())
        # Características fijas: se mantienen en todos los períodos.
        # Haremos que sean las que no son outcomes
        self.fixed_features = [
            var for var in config['variables'].keys()
            if var not in self.outcomes
        ]

    def export_panel_and_config(self, exclude_unobs: bool = True):
        if not hasattr(self, 'panel'):
            raise ValueError("Simulación no ejecutada. Llama a simulate() antes de exportar.")

        panel = self.panel
        if exclude_unobs:
            unobs = [var for var, spec in self.features.items() if not spec['observable']]
            cols = [c for c in panel.columns if c not in unobs]
            export_df = panel[cols]
        else:
            export_df = panel

        export_df = self._order_panel_columns(export_df)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(DATA_DIR, f"simulacion_{timestamp}")
        os.makedirs(base_path, exist_ok=True)

        panel_path = os.path.join(base_path, f"panel.csv")
        export_df.to_csv(panel_path, index=False)

        config_path = os.path.join(base_path, f"config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

        print(f"Exportado: {panel_path}")
        print(f"Exportado: {config_path}")

    def _generate_variable(self, spec: Dict, n: int) -> np.ndarray:
        """Genera variable según especificación."""
        dist, params = spec['distribution'], spec['params']

        if dist == 'normal':
            values = self.rng.normal(params.get('mean', 0), params.get('sd', 1), n)
        elif dist == 'lognormal':
            values = self.rng.lognormal(params.get('mean', 0), params.get('sigma', 1), n)
        elif dist == 'exponential':
            values = self.rng.exponential(params.get('scale', 1), n)
        elif dist == 'bernoulli':
            values = self.rng.binomial(1, params.get('p', 0.5), n)
        elif dist == 'beta':
            values = self.rng.beta(params.get('a', 2), params.get('b', 2), n)
        elif dist == 'categorical':
            values = self.rng.choice(params['categories'], n, p=params.get('probs'))
        else:
            raise ValueError(f"Distribución desconocida: {dist}")

        if spec.get('es_log'):
            values = np.exp(values)

        if 'min' in spec:
            values = np.maximum(values, spec['min'])
        if 'max' in spec:
            values = np.minimum(values, spec['max'])
        if spec.get('integer'):
            values = np.round(values).astype(int)

        return values

    def _generate_initial_conditions(self) -> pd.DataFrame:
        """
        Genera características iniciales (t = 0) de todas las empresas de
        acuerdo a la distribución especificada en la configuración.
        """
        n = self.config['n_empresas']
        # Columna "id_firma" con IDs únicos para cada empresa
        data = pd.DataFrame({'id_firma': range(n)})

        for var_name, spec in self.config['variables'].items():
            values = self._generate_variable(spec, n)
            data[f'{var_name}_0'] = values

        # inicio_firma: período en que cada firma empieza a existir.
        # Se sortea uniformemente entre 0 y (t0 - min_margin), garantizando
        # que toda firma tenga al menos min_periodos_pre_programa períodos
        # de historia antes del inicio del programa.
        t0 = self.config['periodo_inicio_programa']
        min_margin = self.config.get('min_periodos_pre_programa', 1)
        data['inicio_firma'] = self.rng.integers(0, t0 - min_margin + 1, size=n)

        return data

    def _evolve_outcome(
        self,
        firms: pd.DataFrame,
        prev_values: np.ndarray,
        outcome: str,
        t: int,
        treatment_effect: np.ndarray = None
    ) -> np.ndarray:
        """
        Evoluciona un outcome de t-1 a t según el modelo dinámico.

        Y_t = rho*Y_{t-1} + Σ_k [ β_k · X(i,k) ] + shock (ruido) + efecto_tratamiento

        Args:
            prev_values: Valores en t-1
            outcome: Nombre del outcome
            t: Período actual
            treatment_effect: Efecto del tratamiento a agregar

        Returns:
            Valores en t
        """
        dyn = self.config['dinamica_outcomes'][outcome]

        # Efecto de otras variables sobre el crecimiento
        other_vars_effect = 0
        for var, coef in dyn.get('efectos_variables', {}).items():
            col = f'{var}_{t-1}' if f'{var}_{t-1}' in firms.columns else f'{var}_0'
            if col in firms.columns:
                if self.config['variables'][var]['distribution'] == 'categorical':
                    # Para variables categóricas, el efecto es por categoría
                    for category, cat_effect in coef.items():
                        other_vars_effect += np.where(firms[col] == category, cat_effect, 0)
                else:
                    other_vars_effect += coef * firms[col].values

        # Variable continua: AR(1) con ruido
        n = len(prev_values)
        shock = self.rng.normal(0, dyn.get('volatilidad', 0), n)
        new_values = dyn['persistencia'] * prev_values + other_vars_effect + shock

        if treatment_effect is not None:
            if dyn['efecto_tratamiento'] == 'porcentual':
                new_values = new_values * (1 + treatment_effect)
            else:
                new_values = new_values + treatment_effect

        new_values = np.maximum(new_values, dyn.get('min', 0))

        # Asegurar tipos
        if dyn.get('integer'):
            new_values = np.round(new_values).astype(int)

        # Asegurar rangos
        if dyn.get('min'):
            new_values = np.maximum(new_values, dyn['min'])

        return new_values

    def _compute_treatment_effect(
        self,
        t: int,
        outcome: str,
        firms: pd.DataFrame
    ) -> np.ndarray:
        """
        Calcula efecto dinámico del tratamiento sobre una variable
        específica (outcome).

        tau_i(k) = (tau_imm + tau_grad * min(k, k_max)) * (1 + heterogeneidad)

        Args:
            t (int): Período actual.
            outcome (str): Nombre del outcome sobre la cual aplicar el efecto.
            firms (pd.DataFrame): DataFrame con características de las empresas.
        """
        ef = self.config['efectos_tratamiento'][outcome]
        n = len(firms)

        # Períodos desde tratamiento
        periods_since = np.where(
            firms['periodo_tratamiento'] >= 0,
            t - firms['periodo_tratamiento'],
            -999
        )

        is_treated = periods_since >= 0
        # Máxima cantidad de períodos para efecto gradual del tratamiento
        time_factor = np.clip(periods_since, 0, ef['periodos_hasta_maximo'])

        base = ef['efecto_inmediato'] + ef['efecto_gradual'] * time_factor
        base = np.minimum(base, ef['efecto_maximo'])
        base = np.where(is_treated, base, 0)

        # Heterogeneidad
        het = np.zeros(n)
        for var, coef in ef.get('heterogeneidad', {}).items():
            col = f'{var}_0' if f'{var}_0' in firms.columns else var
            if col in firms.columns:
                het += coef * np.clip(firms[col].values, -2, 2)

        het = np.clip(het, -0.4, 0.4)
        total = base * (1 + het)

        return np.where(is_treated, total, 0)

    def _check_eligibility(self, firms: pd.DataFrame, t: int) -> pd.Series:
        """
        Evalúa elegibilidad en período t utilizando criterios determinísticos.
        Todos los criterios tienen que ser cumplidos (AND) para que una empresa
        sea elegible.

        IMPORTANTE: para elegibilidad, solo se consideran las variables
            observables en t.

        Args:
            firms (pd.DataFrame): DataFrame con características de las empresas.
            t (int): Período actual.

        Returns:
            Serie booleana indicando elegibilidad
        """
        eligible = pd.Series(True, index=firms.index)

        for rule in self.config['elegibilidad']:
            var = rule['variable']
            col = f'{var}_{t}' if f'{var}_{t}' in firms.columns else f'{var}_0'
            if col not in firms.columns:
                continue

            ops = {
                'ge': lambda x, v: x >= v,
                'le': lambda x, v: x <= v,
                'gt': lambda x, v: x > v,
                'lt': lambda x, v: x < v,
            }
            eligible = eligible & ops[rule['operator']](firms[col], rule['value'])

        return eligible

    def _compute_demonstration_effect(self, firms: pd.DataFrame, t: int) -> float:
        """
        Calcula efecto demostración basado en resultados observados de cohortes
        previas. El "efecto de demostración" significa que, cuando a las empresas
        ya tratadas les va bien, eso hace que las empresas aún no tratadas estén
        más propensas a entrar al programa en las cohortes siguientes.

        Lógica:
        - Observa el crecimiento promedio de empleados en empresas ya tratadas
        - Compara con su baseline inicial
        - Aplica decaimiento temporal según tiempo desde tratamiento
        - Añade ruido aleatorio
        """
        demo_config = self.config['efecto_demostracion']

        # Si está desactivado, retornar 0
        if not demo_config.get('activado', False):
            return 0.0

        # Identificar empresas ya tratadas
        # NOTA: toma TODAS las ya tratadas, no solo las de la cohorte inmediata previa
        treated_mask = firms['tratado'] & (firms['periodo_tratamiento'] < t)

        if not treated_mask.any():
            return 0.0  # No hay cohortes previas para observar

        treated_firms = firms[treated_mask]

        periods_since = t - treated_firms['periodo_tratamiento'].values
        decay_weights = demo_config['decaimiento'] ** periods_since

        # Crecimiento relativo desde entrada al programa, normalizado por efecto_maximo
        # de cada outcome para que sean comparables entre sí
        outcome_growths = []
        for outcome in self.outcomes:
            current_col = f'{outcome}_{t}'
            if current_col not in firms.columns:
                continue

            efecto_maximo = self.config['efectos_tratamiento'][outcome]['efecto_maximo']

            current_values = treated_firms[current_col].values
            baseline_values = np.array([
                firms.loc[idx, f'{outcome}_{pt}']
                for idx, pt
                in zip(treated_firms.index, treated_firms['periodo_tratamiento'].values)
            ])
            growth = (current_values - baseline_values) / (np.abs(baseline_values) + 1)
            normalized_growth = growth / efecto_maximo
            outcome_growths.append(np.average(normalized_growth, weights=decay_weights))

        if not outcome_growths:
            return 0.0

        weighted_growth = np.mean(outcome_growths)

        # Calcular efecto con sensibilidad
        base_effect = demo_config['sensibilidad'] * weighted_growth

        # Añadir ruido
        noise = self.rng.normal(0, demo_config['ruido'])
        total_effect = base_effect + noise

        # Limitar el efecto
        total_effect = np.clip(total_effect, -0.5, 1.0)

        return total_effect

    def _compute_propensity(
        self,
        firms: pd.DataFrame,
        t: int,
        demo_effect: float = 0.0
    ) -> pd.Series:
        """
        Calcula la probabilidad de que cada empresa aplique al programa
        (propensity score) en el período t, considerando el efecto demostración.

        Args:
            firms (pd.DataFrame): DataFrame con características de las empresas.
            t (int): Período actual.
            demo_effect (float): Efecto demostración a incorporar en el cálculo.

        Returns:
            pd.Series: Serie con la probabilidad de aplicar para cada empresa.
        """
        sel = self.config['seleccion']
        n = len(firms)

        z = np.full(n, sel['intercepto_base'] + demo_effect)

        for var, coef in sel['efectos_variables'].items():
            # Notar que los valores de las variables para la selección se toman
            # del período actual (t) si existen, sino del período 0
            col = f'{var}_{t}' if f'{var}_{t}' in firms.columns else f'{var}_0'
            if col in firms.columns and np.issubdtype(firms[col].dtype, np.number):
                z = z + coef * firms[col].values

        if 'sector_0' in firms.columns:
            for sector, effect in sel.get('efectos_sector', {}).items():
                z[firms['sector_0'] == sector] += effect

        if 'region_0' in firms.columns:
            for region, effect in sel.get('efectos_region', {}).items():
                z[firms['region_0'] == region] += effect

        z = np.clip(z, -10, 10)  # Evitar overflow
        return pd.Series(expit(z), index=firms.index)

    def _assign_treatment(
        self,
        firms: pd.DataFrame,
        propensity: pd.Series,
        eligible: pd.Series,
        cupo: int,
    ) -> tuple[pd.Series, pd.Index]:
        """
        Asigna tratamiento con cupo.

        1. Cada elegible decide si aplica (Bernoulli con p = propensity)
        2. Si aplicantes > cupo, se seleccionan los de mayor score

        Returns:
            pd.Series: Serie booleana indicando qué empresas resultan tratadas
                en el período actual.
            pd.Index: Índice de empresas que resultaron controles.
        """
        tratado = pd.Series(False, index=firms.index)

        # Los candidatos son los que cumplen con las condiciones (determinísticas)
        # para participar y no han sido tratados antes
        already_treated = firms['tratado']
        candidates = eligible & ~already_treated

        if candidates.sum() == 0:
            return tratado, pd.Index([])

        candidate_idx = firms.index[candidates]
        # self.rng.random(len(candidate_idx)): genera un número entre 0 y 1 por
        # candidato.
        # Un candidato aplica cuando el número aleatorio es menor a su propensity
        # Es decir, si su propensity es alta, tiene más chances de aplicar
        applies = self.rng.random(len(candidate_idx)) < propensity[candidate_idx].values
        applicant_idx = candidate_idx[applies]

        if len(applicant_idx) == 0:
            return tratado, pd.Index([])

        # Split 50/50 aleatorio
        shuffled = self.rng.permutation(applicant_idx)
        mid = len(shuffled) // 2
        treated_idx = pd.Index(shuffled[:mid])
        control_idx = pd.Index(shuffled[mid:])

        if len(treated_idx) > cupo:
            # Si no hay cupo para todos, seleccionar en base a scores. El score
            # es simplemente sumar un poco de ruido al propensity para
            # introducir aleatoriedad en la selección
            scores = propensity[treated_idx].values + self.rng.normal(0, 0.1, len(treated_idx))
            treated_idx = treated_idx[np.argsort(-scores)[:cupo]]

        return treated_idx, control_idx

    def _order_panel_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ordena columnas del DataFrame.
        """
        first_cols = [
            'id_firma',
            'inicio_firma',
            't',
            'tratado_en_t',
            'control_en_t',
        ]
        existing_first_cols = [c for c in first_cols if c in df.columns]
        remaining_cols = [c for c in df.columns if c not in existing_first_cols]
        return df[existing_first_cols + remaining_cols]

    def simulate(self) -> PanderaDataFrame[PanelSchema]:
        """
        Ejecuta la simulación completa del panel.

        Returns:
            DataFrame en formato long (firma_id × t), donde cada firma
            tiene filas solo desde su inicio_firma en adelante.
        """
        n_periods = self.config['n_periodos']
        n_cohorts = self.config['n_cohortes']
        t0 = self.config['periodo_inicio_programa']

        # 1. Generar condiciones iniciales Los valores inicial de cada una de
        # las variables se generan según la distribución especificada en la
        # configuración
        firms = self._generate_initial_conditions()
        firms['tratado'] = False
        firms['cohorte'] = -1
        firms['periodo_tratamiento'] = -1

        # Dataset final
        panel_data = []

        # 2. Simular cada período
        for t in range(n_periods):
            # pdata = period data: DataFrame temporal para almacenar resultados del
            # período t antes de agregarlos al panel final
            pdata = firms[['id_firma', 'inicio_firma']].copy()
            pdata['t'] = t

            # Para las features que son fijas, copiamos el valor del período 0
            for var in self.fixed_features:
                if f'{var}_0' in firms.columns:
                    pdata[var] = firms[f'{var}_0']

            # 3. Evolucionar outcomes
            for outcome in self.outcomes:
                if t == 0:
                    new_values = firms[f'{outcome}_0'].values.copy()
                else:
                    prev_values = firms[f'{outcome}_{t-1}'].values
                    te = self._compute_treatment_effect(t, outcome, firms)
                    new_values = self._evolve_outcome(firms, prev_values, outcome, t, te)

                # Estos values ya tienen el efecto aplicado del tratamiento a
                # aquellos que les corresponde
                pdata[outcome] = new_values
                # Agregamos una columna más a firms
                firms[f'{outcome}_{t}'] = new_values

            # 4. Asignar tratamiento si es período de cohorte
            # t0 es el primer período de tratamiento
            control_this_period = pd.Series(False, index=firms.index)
            cohort_in_period = t - t0   # Indexado en 0: cohorte 0 en t0, cohorte 1 en t0+1, etc.
            if 0 <= cohort_in_period < n_cohorts:   # Si el t actual corresponde a una cohorte
                # En base a los valores generados para el período actual, verificamos
                # elegibilidad para el tratamiento en esta cohorte (condiciones
                # determinísticas)
                eligible = self._check_eligibility(firms, t)

                # Calcular efecto demostración: cuánto afecta el desempeño de
                # cohortes anteriores a la probabilidad de entrar al programa
                demo_effect = self._compute_demonstration_effect(firms, t)

                # Calcular propensity con efecto demostración
                propensity = self._compute_propensity(firms, t, demo_effect=demo_effect)

                cupo = self.cupos[cohort_in_period]
                treated_idx, control_idx = self._assign_treatment(
                    firms, propensity, eligible, cupo
                )

                # Tratados primero — tienen prioridad
                firms.loc[treated_idx, 'tratado'] = True
                firms.loc[treated_idx, 'cohorte'] = cohort_in_period
                firms.loc[treated_idx, 'periodo_tratamiento'] = t

                # Controles
                control_this_period.loc[control_idx] = True

                demo_msg = f", efecto demo: {demo_effect:+.3f}" if demo_effect != 0 else ""
                print(
                    f"  Período {t} (Cohorte {cohort_in_period}): "
                    f"{len(treated_idx)} tratadas (cupo: {cupo}{demo_msg})"
                )

            # 5. Agregar estado de tratamiento
            pdata['tratado_en_t'] = firms['periodo_tratamiento'] == t
            pdata['control_en_t'] = control_this_period

            panel_data.append(pdata)

        # 6. Combinar y recortar: cada firma aparece solo desde su inicio_firma
        panel = pd.concat(panel_data, ignore_index=True)
        panel = panel[panel['t'] >= panel['inicio_firma']].reset_index(drop=True)
        panel = panel.sort_values(['id_firma', 't']).reset_index(drop=True)

        # 7. Firmas que fueron control en cohorte previa y luego se trataron:
        # pierden condición de control en todos sus períodos
        treated_ids = firms.loc[firms['tratado'], 'id_firma'].values
        panel.loc[panel['id_firma'].isin(treated_ids), 'control_en_t'] = False

        panel = self._order_panel_columns(panel)

        PanelSchema.validate(panel)
        self.panel = panel
        return panel
