from data_config import DATA_CONFIG
from simulator import DataSimulator

def main():
    print(f"Configuración: {DATA_CONFIG['nombre_programa']}")
    print(f"  Empresas: {DATA_CONFIG['n_empresas']:,}")
    print(f"  Períodos: {DATA_CONFIG['n_periodos']}")
    print(f"  Inicio programa: Período {DATA_CONFIG['periodo_inicio_programa']}")
    print(f"  Cohortes: {DATA_CONFIG['n_cohortes']} con cupos {DATA_CONFIG['cupo_por_cohorte']}")

    simulator = DataSimulator(DATA_CONFIG)
    panel = simulator.simulate()

    # Resumen
    treated_ids = set(panel.loc[panel['tratado_en_t'], 'id_firma'])
    control_ids = set(panel.loc[panel['control_en_t'], 'id_firma']) - treated_ids
    n_treated = len(treated_ids)
    n_control = len(control_ids)
    n_nini = DATA_CONFIG['n_empresas'] - n_treated - n_control

    print(f"\nESTRUCTURA DEL PANEL")
    print("=" * 70)
    print(f"Dimensiones: {panel.shape[0]:,} obs ({DATA_CONFIG['n_empresas']:,} × {DATA_CONFIG['n_periodos']} períodos)")
    print(f"Empresas tratadas: {n_treated}")
    print(f"Empresas control:  {n_control}")
    print(f"Empresas NiNi:     {n_nini}")

    simulator.export_panel_and_config(exclude_unobs=True)

if __name__ == "__main__":
    main()