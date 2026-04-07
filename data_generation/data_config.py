DATA_CONFIG = {
    # -------------------------------------------------------------------------
    # PARÁMETROS GENERALES
    # -------------------------------------------------------------------------
    'nombre_programa': 'Programa de Crédito Productivo PyME',
    'n_empresas': 1000,
    'random_seed': 2024,

    # -------------------------------------------------------------------------
    # ESTRUCTURA TEMPORAL (PANEL)
    # -------------------------------------------------------------------------
    'n_periodos': 12,                 # Total de períodos
    'periodo_inicio_programa': 5,     # El programa comienza en t=5
    'min_periodos_pre_programa': 3,   # Mínimo de períodos que una firma existe antes del programa

    # -------------------------------------------------------------------------
    # COHORTES Y CUPOS
    # -------------------------------------------------------------------------
    'n_cohortes': 4,
    'cupo_por_cohorte': [80, 100, 120, 150],

    # -------------------------------------------------------------------------
    # CARACTERÍSTICAS DE LAS EMPRESAS
    # -------------------------------------------------------------------------
    'variables': {
        'empleados': {
            'distribution': 'lognormal',
            'params': {'mean': 2.3, 'sigma': 0.9},
            'min': 1, 'max': 500,
            'integer': True,
            'observable': True,
        },
        'salario_promedio': {
            'distribution': 'lognormal',
            'params': {'mean': 12.0, 'sigma': 0.3},
            'min': 80000, 'max': 800000,
            'observable': True,
            'es_log': True,
        },
        'tiene_credito': {
            'distribution': 'bernoulli',
            'params': {'p': 0.30},
            'observable': True,
        },
        'antiguedad': {
            'distribution': 'exponential',
            'params': {'scale': 8},
            'min': 0.5, 'max': 50,
            'observable': True,
        },
        'sector': {
            'distribution': 'categorical',
            'params': {
                'categories': ['manufactura', 'comercio', 'servicios', 'tecnologia'],
                'probs': [0.30, 0.35, 0.25, 0.10]
            },
            'observable': True,
        },
        'region': {
            'distribution': 'categorical',
            'params': {
                'categories': ['centro', 'norte', 'sur', 'litoral'],
                'probs': [0.45, 0.20, 0.20, 0.15]
            },
            'observable': True,
        },
        'exportadora': {
            'distribution': 'bernoulli',
            'params': {'p': 0.12},
            'observable': True,
        },
        'ratio_formalidad': {
            'distribution': 'beta',
            'params': {'a': 4, 'b': 2},
            'observable': True,
        },
        # Variables NO observables
        'calidad_gerencial': {
            'distribution': 'normal',
            'params': {'mean': 0, 'sd': 1},
            'observable': False,
        },
        'productividad_latente': {
            'distribution': 'normal',
            'params': {'mean': 0, 'sd': 1},
            'observable': False,
        },
        'propension_credito': {
            'distribution': 'normal',
            'params': {'mean': 0, 'sd': 1},
            'observable': False,
        },
    },

    # -------------------------------------------------------------------------
    # DINÁMICA TEMPORAL DE LOS OUTCOMES
    # -------------------------------------------------------------------------
    # efectos_variables: cómo cada variable afecta la evolución de cada outcome
    # (se multiplica por el valor de la variable). Debería haber una entrada por
    # cada variable que NO sea outcome.
    'dinamica_outcomes': {
        'empleados': {
            'persistencia': 0.95,
            'volatilidad': 0.05,
            'integer': True,
            'min': 1,
            'efecto_tratamiento': 'aditivo',
            'efectos_variables': {
                # Firmas más antiguas tienen estructuras de RRHH más
                # consolidadas (+1.2 emp en LR por año de edad)
                'antiguedad': 0.06,
                # Exportadoras tienen demanda externa sostenida que impulsa
                # contratación (+16 emp en LR)
                'exportadora': 0.8,
                # Mayor formalidad refleja capacidad institucional de crecer con
                # personal estable (+10 emp en LR)
                'ratio_formalidad': 0.5,
                # Mejor gestión → mayor capacidad de coordinar equipos más
                # grandes (+6 emp en LR por 1-sigma)
                'calidad_gerencial': 0.3,
                # Alta productividad latente permite escalar operaciones con más
                # personal (+8 emp en LR)
                'productividad_latente': 0.4,
                # Firmas con mayor propensión al crédito invierten más y
                # contratan (+3 emp en LR)
                'propension_credito': 0.15,
            }
        },
        'salario_promedio': {
            'persistencia': 0.98,
            'volatilidad': 0.025,
            'min': 50000,
            'efecto_tratamiento': 'porcentual',
            'efectos_variables': {
                # Firmas más antiguas tienen escalafones salariales más
                # desarrollados (LR: ~10k$ por año de edad)
                'antiguedad': 25,
                # Exportadoras pagan salarios premium para atraer talento con
                # competencias internacionales (LR: ~10k$)
                'exportadora': 200,
                # Mayor formalidad implica cumplir convenios colectivos y pisos
                # salariales (LR: ~5k$)
                'ratio_formalidad': 100,
                # Buena gestión retiene personal clave con salarios más
                # competitivos (LR: ~4k$ por 1-sigma)
                'calidad_gerencial': 80,
                # Mayor productividad latente se traslada a salarios vía
                # negociación (LR: ~5k$ por 1-sigma)
                'productividad_latente': 100,
                # Acceso fluido al crédito permite sostener nóminas más altas
                # sin tensión de caja (LR: ~2.5k$)
                'propension_credito': 50,
            }
        },
        'tiene_credito': {
            'persistencia': 0.90,
            'tendencia_base': 0.01,
            'efecto_ciclo': 0.015,
            'es_binaria': True,
            # NOTA: el modelo binario actual no aplica efectos_variables (ver _evolve_outcome, rama es_binaria).
            'efectos_variables': {
                # Firmas más antiguas tienen historial crediticio y mayor
                # confianza bancaria
                'antiguedad': 0.005,
                # Exportadoras tienen flujos en divisas que facilitan el acceso
                # al crédito formal
                'exportadora': 0.08,
                # Alta formalidad es requisito habitual para acceder al sistema
                # bancario
                'ratio_formalidad': 0.12,
                # Buena gestión → mejores garantías y proyectos más financiables
                'calidad_gerencial': 0.04,
                # Productividad latente alta señaliza solvencia ante los bancos
                'productividad_latente': 0.03,
                # Variable construida para reflejar exactamente esta propensión
                'propension_credito': 0.10,
            }
        },
    },

    # -------------------------------------------------------------------------
    # REGLAS DE ELEGIBILIDAD
    # -------------------------------------------------------------------------
    # Se deben cumplir todas las condiciones para ser elegible
    # NOTAR que son reglas determinísticas
    'elegibilidad': [
        {'variable': 'empleados', 'operator': 'le', 'value': 100},
        {'variable': 'empleados', 'operator': 'ge', 'value': 3},
        {'variable': 'antiguedad', 'operator': 'ge', 'value': 1},
        {'variable': 'ratio_formalidad', 'operator': 'ge', 'value': 0.5},
    ],

    # -------------------------------------------------------------------------
    # MODELO DE SELECCIÓN
    # -------------------------------------------------------------------------
    'seleccion': {
        'intercepto_base': -1.5,
        # Coeficientes para variables al calcular la probabilidad de participar
        # en el programa (se multiplican por la variable correspondiente)
        'efectos_variables': {
            'empleados': 0.008,
            'antiguedad': 0.015,
            'exportadora': 0.35,
            'tiene_credito': 0.25,
            'ratio_formalidad': 0.4,
            'calidad_gerencial': 0.5,       # NO OBSERVABLE
            'productividad_latente': 0.25,  # NO OBSERVABLE
            'propension_credito': 0.6,      # NO OBSERVABLE
        },
        # Cómo pertencer a un sector afecta la probabilidad de participar
        # Por ejemplo: pertenecer al sector 'comercio' reduce la probabilidad en 15%
        'efectos_sector': {
            'manufactura': 0.0,
            'comercio': -0.15,
            'servicios': -0.1,
            'tecnologia': 0.3,
        },
        # Cómo la región afecta la probabilidad de participar
        # Por ejemplo: estar en la región 'norte' reduce la probabilidad en 25%
        'efectos_region': {
            'centro': 0.0,
            'norte': -0.25,
            'sur': -0.15,
            'litoral': -0.1,
        },
    },

    # -------------------------------------------------------------------------
    # EFECTO DEMOSTRACIÓN
    # -------------------------------------------------------------------------
    'efecto_demostracion': {
        'activado': True,
        'sensibilidad': 0.3,
        'decaimiento': 0.7,
        'ruido': 0.15,
    },

    # -------------------------------------------------------------------------
    # EFECTOS DEL TRATAMIENTO (DINÁMICOS)
    # -------------------------------------------------------------------------
    'efectos_tratamiento': {
        'empleados': {
            'efecto_inmediato': 1.0,
            'efecto_gradual': 0.5,
            'efecto_maximo': 5.0,
            'periodos_hasta_maximo': 6,
            'heterogeneidad': {'calidad_gerencial': 0.3},
        },
        'salario_promedio': {
            'efecto_inmediato': 0.015,      # 1.5% inmediato
            'efecto_gradual': 0.008,        # 0.8% por período
            'efecto_maximo': 0.08,          # Máximo 8%
            'periodos_hasta_maximo': 8,
            'heterogeneidad': {'calidad_gerencial': 0.01},
        },
        'tiene_credito': {
            'efecto_inmediato': 0.30,
            'efecto_gradual': 0.04,
            'efecto_maximo': 0.45,
            'periodos_hasta_maximo': 4,
            'heterogeneidad': {'propension_credito': 0.08},
        },
    },

    # -------------------------------------------------------------------------
    # CICLO ECONÓMICO
    # -------------------------------------------------------------------------
    'ciclo_economico': {
        'activado': True,
        'volatilidad_agregada': 0.015,
        'shocks_por_periodo': {},
    },
}
