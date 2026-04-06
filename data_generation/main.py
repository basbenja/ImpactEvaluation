from data_config import DATA_CONFIG
from data_generation.simulator import DataSimulator

def main():
    print(f"Configuración: {DATA_CONFIG['nombre_programa']}")
    print(f"  Empresas: {DATA_CONFIG['n_empresas']:,}")
    print(f"  Períodos: {DATA_CONFIG['n_periodos']}")
    print(f"  Inicio programa: Período {DATA_CONFIG['periodo_inicio_programa']}")
    print(f"  Cohortes: {DATA_CONFIG['n_cohortes']} con cupos {DATA_CONFIG['cupo_por_cohorte']}")

    simulator = DataSimulator(DATA_CONFIG)
    panel = simulator.simulate()

    # Resumen
    treated = panel.groupby('id_firma')['tratado'].max()
    n_treated = treated.sum()
    n_control = (panel.groupby('id_firma')['control'].max() & ~treated).sum()

    print(f"\nESTRUCTURA DEL PANEL")
    print("=" * 70)
    print(f"Dimensiones: {panel.shape[0]:,} obs ({DATA_CONFIG['n_empresas']:,} × {DATA_CONFIG['n_periodos']} períodos)")
    print(f"Empresas tratadas: {n_treated}")
    print(f"Empresas control: {n_control}")

    simulator.export_panel_and_config(exclude_unobs=True)

if __name__ == "__main__":
    main()