import numpy as np
import pandas as pd

def generate_panel(
    n_firms: int = 200,
    n_periods: int = 20,
    cohort_starts: list[int] = [8, 9, 10],
    n_treated_per_cohort: int = 20,
    n_control_per_cohort: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera un panel sintético simple con:
    - Firms con distintos períodos de inicio (proxy de antigüedad)
    - Tratados y controles por cohorte
    - NiNi: nunca tratados ni controles
    """
    rng = np.random.default_rng(seed)

    # ── 1. Asignar período de inicio a cada firma ──────────────────────
    # Cada firma "nace" en un período aleatorio entre 0 y n_periods // 2
    firm_ids = np.arange(n_firms)
    firm_start = rng.integers(0, n_periods // 2, size=n_firms)  # período de inicio

    # ── 2. Asignar roles: T, C, NiNi ──────────────────────────────────
    roles = pd.DataFrame({'firm_id': firm_ids, 'firm_start': firm_start})
    roles['tratado'] = False
    roles['control'] = False
    roles['cohort']  = -1

    available = firm_ids.copy()
    rng.shuffle(available)

    idx = 0
    for cohort_id, t_k in enumerate(cohort_starts):
        # Tratados de esta cohorte
        t_firms = available[idx: idx + n_treated_per_cohort]
        roles.loc[roles['firm_id'].isin(t_firms), 'tratado'] = True
        roles.loc[roles['firm_id'].isin(t_firms), 'cohort']  = cohort_id
        idx += n_treated_per_cohort

        # Controles de esta cohorte
        c_firms = available[idx: idx + n_control_per_cohort]
        roles.loc[roles['firm_id'].isin(c_firms), 'control'] = True
        roles.loc[roles['firm_id'].isin(c_firms), 'cohort']  = cohort_id
        idx += n_control_per_cohort

    # ── 3. Generar filas del panel ─────────────────────────────────────
    rows = []
    for _, firm in roles.iterrows():
        firm_id    = firm['firm_id']
        start      = firm['firm_start']
        is_treated = firm['tratado']
        is_control = firm['control']
        cohort_id  = int(firm['cohort'])

        # Período de tratamiento/control de referencia
        t_k = cohort_starts[cohort_id] if cohort_id >= 0 else None

        # Generar valores base para y_1 e y_2
        y1_base = rng.normal(10, 2)
        y2_base = rng.normal(50, 10)

        for t in range(start, n_periods):
            # Solo períodos desde que la firma "existe"
            y1 = y1_base + rng.normal(0, 0.5) + 0.1 * t
            y2 = y2_base + rng.normal(0, 2)   + 0.2 * t

            # Efecto de tratamiento post t_k (solo para tratados)
            if is_treated and t_k is not None and t >= t_k:
                y1 += 2.0
                y2 += 5.0

            rows.append({
                'firm_id': firm_id,
                't':       t,
                'y_1':     round(y1, 3),
                'y_2':     round(y2, 3),
                'tratado': is_treated,
                'control': is_control,
                'cohort':  cohort_id,
            })

    panel = pd.DataFrame(rows).sort_values(['firm_id', 't']).reset_index(drop=True)
    return panel


if __name__ == '__main__':
    panel = generate_panel()

    print(f"Shape: {panel.shape}")
    print(f"Firms: {panel['firm_id'].nunique()}")
    print(f"\nDistribución de roles (a nivel firma):")
    firm_status = panel.groupby('firm_id').last()[['tratado', 'control']]
    print(f"  Tratados : {firm_status['tratado'].sum()}")
    print(f"  Controles: {(~firm_status['tratado'] & firm_status['control']).sum()}")
    print(f"  NiNi     : {(~firm_status['tratado'] & ~firm_status['control']).sum()}")
    print(f"\nObservaciones por firma (min/max):")
    obs_per_firm = panel.groupby('firm_id').size()
    print(f"  min: {obs_per_firm.min()} | max: {obs_per_firm.max()}")
    print(f"\nMuestra:")
    print(panel.head(10).to_string())

    panel.to_csv('panel_simple.csv', index=False)
    print("\nGuardado en panel_simple.csv")
