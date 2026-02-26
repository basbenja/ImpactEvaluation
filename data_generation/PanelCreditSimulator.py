import json
import os
import numpy as np
import pandas as pd
import sys

from datetime import datetime
# The expit function, also known as the logistic sigmoid function, is defined as
# expit(x) = 1/(1+exp(-x)).
# Es la sigmoide básicamente, le pasas cualquier número y te devuelve un valor
# entre 0 y 1, imitando una probabilidad.
from scipy.special import expit
from typing import Dict

sys.path.append('..')
from config import DATA_DIR

class PanelCreditSimulator:
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
        self.fixed_features = [
            var for var in config['variables'].keys()
            if var not in self.outcomes
        ]

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
        # Columna "firm_id" con IDs únicos para cada empresa
        data = pd.DataFrame({'firm_id': range(n)})

        for var_name, spec in self.config['variables'].items():
            values = self._generate_variable(spec, n)
            data[f'{var_name}_0'] = values

        return data

    def _evolve_outcome(
        self,
        prev_values: np.ndarray,
        outcome: str,
        t: int,
        treatment_effect: np.ndarray = None
    ) -> np.ndarray:
        """
        Evoluciona un outcome de t-1 a t según el modelo dinámico.

        Para continuas: Y_t = Y_{t-1} * (1 + tendencia + ciclo + shock) + efecto_tratamiento
        Para binarias: P(Y_t=1) = rho*Y_{t-1} + (1-rho)*p_base + ciclo + efecto_tratamiento

        Args:
            prev_values: Valores en t-1
            outcome: Nombre del outcome ('empleados', 'salario_promedio', 'tiene_credito')
            t: Período actual
            treatment_effect: Efecto del tratamiento a agregar

        Returns:
            Valores en t
        """
        dyn = self.config['dinamica_outcomes'][outcome]
        n = len(prev_values)

        if dyn.get('es_binaria'):
            # Variable binaria: modelo de transición
            prob = dyn['persistencia'] * prev_values + (1 - dyn['persistencia']) * 0.3
            prob += dyn.get('efecto_ciclo', 0) * self.aggregate_shocks[t]

            if treatment_effect is not None:
                prob = prob + treatment_effect

            prob = np.clip(prob, 0.02, 0.98)
            return self.rng.binomial(1, prob).astype(float)

        # Variable continua: crecimiento porcentual
        shock = self.rng.normal(0, dyn['volatilidad'], n)
        cycle = dyn.get('efecto_ciclo', 0) * self.aggregate_shocks[t]
        growth = dyn['tendencia_base'] + shock + cycle

        new_values = prev_values * (1 + growth)

        if treatment_effect is not None:
            if outcome == 'salario_promedio':
                # Para salario: efecto multiplicativo (porcentual)
                new_values = new_values * (1 + treatment_effect)
            else:
                # Para empleados: efecto aditivo
                new_values = new_values + treatment_effect

        new_values = np.maximum(new_values, dyn.get('min', 0))

        if dyn.get('integer'):
            new_values = np.round(new_values).astype(int)

        return new_values

    def _compute_treatment_effect(
        self,
        outcome: str,
        periods_since: np.ndarray,
        firms: pd.DataFrame
    ) -> np.ndarray:
        """
        Calcula efecto dinámico del tratamiento sobre una variable
        específica (outcome).

        tau_i(k) = (tau_imm + tau_grad * min(k, k_max)) * (1 + heterogeneidad)

        Args:
            outcome (str): Nombre del outcome sobre la cual aplicar el efecto.
            periods_since (np.ndarray): Array que contiene por cada empresa, el
                número de períodos desde que recibió el tratamiento.
            firms (pd.DataFrame): DataFrame con características de las empresas.
        """
        ef = self.config['efectos_tratamiento'][outcome]
        n = len(firms)

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
        previas. El “efecto de demostración” significa que, cuando a las empresas
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
        treated_mask = firms['ever_treated'] & (firms['periodo_tratamiento'] < t)

        if not treated_mask.any():
            return 0.0  # No hay cohortes previas para observar

        # Calcular el crecimiento observado en empleados
        treated_firms = firms[treated_mask]

        # Empleados actuales vs iniciales
        current_col = f'empleados_{t-1}' if t > 0 else 'empleados_0'
        if current_col not in firms.columns:
            return 0.0

        current_employees = treated_firms[current_col].values
        initial_employees = treated_firms['empleados_0'].values

        # Crecimiento relativo promedio
        growth = (current_employees - initial_employees) / (initial_employees + 1)

        # Aplicar decaimiento temporal: empresas tratadas hace más tiempo tienen
        # menor efecto
        periods_since = t - treated_firms['periodo_tratamiento'].values
        decay_weights = demo_config['decaimiento'] ** periods_since
        weighted_growth = np.average(growth, weights=decay_weights)

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
        already_treated: pd.Series
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
        treated = pd.Series(False, index=firms.index)

        # Los candidatos son los que cumplen con las condiciones (determinísticas)
        # para participar y no han sido tratados antes
        candidates = eligible & ~already_treated

        if candidates.sum() == 0:
            return treated, pd.Index([])

        candidate_idx = firms.index[candidates]
        # self.rng.random(len(candidate_idx)): genera un número entre 0 y 1 por
        # candidato.
        # Un candidato aplica cuando el número aleatorio es menor a su propensity
        # Es decir, si su propensity es alta, tiene más chances de aplicar
        applies = self.rng.random(len(candidate_idx)) < propensity[candidate_idx].values
        applicant_idx = candidate_idx[applies]

        if len(applicant_idx) == 0:
            return treated, pd.Index([])

        # Split 50/50 aleatorio
        shuffled = self.rng.permutation(applicant_idx)
        mid = len(shuffled) // 2
        treated_idx = pd.Index(shuffled[:mid])
        control_idx = pd.Index(shuffled[mid:])

        if len(treated_idx) <= cupo:
            # Si hay cupo para que todos los aplicantes entren, todos son tratados
            treated[treated_idx] = True
        else:
            # Si no hay cupo para todos, seleccionar en base a scores. El score
            # es simplemente sumar un poco de ruido al propensity para
            # introducir aleatoriedad en la selección
            scores = propensity[treated_idx].values + self.rng.normal(0, 0.1, len(treated_idx))
            selected = treated_idx[np.argsort(-scores)[:cupo]]
            treated[selected] = True

        return treated, control_idx

    def simulate(self) -> pd.DataFrame:
        """
        Ejecuta la simulación completa del panel.

        Returns:
            DataFrame en formato long (firm_id × periodo)
        """
        n_periods = self.config['n_periodos']
        n_cohorts = self.config['n_cohortes']
        t0 = self.config['periodo_inicio_programa']

        # 1. Generar condiciones iniciales
        firms = self._generate_initial_conditions()
        firms['ever_treated'] = False
        firms['cohort'] = -1
        firms['periodo_tratamiento'] = -1
        firms['es_control'] = False

        panel_data = []

        # 2. Simular cada período
        for t in range(n_periods):
            pdata = firms[['firm_id', 'empleados_0', 'salario_promedio_0']].copy()
            pdata['periodo'] = t

            # Copiar características fijas
            for var in self.fixed_features:
                if f'{var}_0' in firms.columns:
                    pdata[var] = firms[f'{var}_0']

            # Períodos desde tratamiento
            periods_since = np.where(
                firms['periodo_tratamiento'] >= 0,
                t - firms['periodo_tratamiento'],
                -999
            )
            pdata['periodos_post_tratamiento'] = np.where(
                periods_since >= 0, periods_since, np.nan
            )

            # 3. Evolucionar outcomes
            for outcome in self.outcomes:
                if t == 0:
                    values = firms[f'{outcome}_0'].values.copy()
                else:
                    prev = firms[f'{outcome}_{t-1}'].values
                    te = self._compute_treatment_effect(outcome, periods_since, firms)
                    values = self._evolve_outcome(prev, outcome, t, te)

                # Asegurar tipos y rangos
                if self.config['dinamica_outcomes'][outcome].get('integer'):
                    values = np.round(values).astype(int)

                if self.config['dinamica_outcomes'][outcome].get('min'):
                    values = np.maximum(values, self.config['dinamica_outcomes'][outcome]['min'])

                # Estos values ya tienen el efecto aplicado del tratamiento a
                # aquellos que les corresponde
                firms[f'{outcome}_{t}'] = values
                pdata[outcome] = values

            # 4. Asignar tratamiento si es período de cohorte
            # t0 es el primer período de tratamiento
            cohort_in_period = t - t0   # Indexado en 0: cohorte 0 en t0, cohorte 1 en t0+1, etc.
            if 0 <= cohort_in_period < n_cohorts:   # Si el t actual corresponde a una cohorte
                cupo = self.cupos[cohort_in_period]
                # En base a los valores generados para el período actual, verificamos
                # elegibilidad para el tratamiento en esta cohorte (condiciones
                # determinísticas)
                eligible = self._check_eligibility(firms, t)

                # Calcular efecto demostración: cuánto afecta el desempeño de
                # cohortes anteriores a la probabilidad de entrar al programa
                demo_effect = self._compute_demonstration_effect(firms, t)

                # Calcular propensity con efecto demostración
                propensity = self._compute_propensity(firms, t, demo_effect=demo_effect)

                treated_now, control_idx = self._assign_treatment(
                    firms, propensity, eligible, cupo, firms['ever_treated']
                )

                # Tratados primero — tienen prioridad
                firms.loc[treated_now, 'control'] = False
                firms.loc[treated_now, 'ever_treated'] = True
                firms.loc[treated_now, 'cohort'] = cohort_in_period
                firms.loc[treated_now, 'periodo_tratamiento'] = t

                # Controles — solo si no son ni fueron tratados
                new_controls = control_idx[~firms.loc[control_idx, 'ever_treated']]
                firms.loc[new_controls, 'control'] = True

                demo_msg = f", efecto demo: {demo_effect:+.3f}" if demo_effect != 0 else ""
                print(f"  Período {t} (Cohorte {cohort_in_period}): {treated_now.sum()} tratadas (cupo: {cupo}{demo_msg})")

            # 5. Agregar estado de tratamiento
            pdata['tratado'] = firms['ever_treated'] & (firms['periodo_tratamiento'] <= t)
            pdata['cohort'] = np.where(pdata['tratado'], firms['cohort'], -1)
            pdata['elegible'] = self._check_eligibility(firms, t)
            pdata['control'] = firms['control']

            panel_data.append(pdata)

        # 6. Combinar y agregar etiquetas temporales
        panel = pd.concat(panel_data, ignore_index=True)
        panel['periodo_relativo'] = panel['periodo'] - t0
        panel['año'] = self.config['año_inicio'] + panel['periodo'] // 4
        panel['trimestre'] = (panel['periodo'] % 4) + 1
        panel['fecha'] = panel['año'].astype(str) + '-Q' + panel['trimestre'].astype(str)

        self.panel = panel

        return panel

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

        timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        base_path = os.path.join(DATA_DIR, f"simulacion_{timestamp}")
        os.makedirs(base_path, exist_ok=True)

        panel_path = os.path.join(base_path, f"panel.csv")
        export_df.to_csv(panel_path, index=False)

        config_path = os.path.join(base_path, f"config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

        print(f"Exportado: {panel_path}")
        print(f"Exportado: {config_path}")