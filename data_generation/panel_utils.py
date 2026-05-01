from pandera.typing import DataFrame

from panel_schema import Col, PanelSchema

def get_treated_per_cohort(panel: DataFrame[PanelSchema]) -> dict[int, list[int]]:
    treated = panel[panel[Col.TRATADO_EN_T]]
    return treated.groupby(Col.T)[Col.ID_FIRMA].apply(list).to_dict()


def get_controls_per_cohort(panel: DataFrame[PanelSchema]) -> dict[int, list[int]]:
    controls = panel[panel[Col.CONTROL_EN_T]]
    return controls.groupby(Col.T)[Col.ID_FIRMA].apply(list).to_dict()
