from pandera.typing import DataFrame

from data_generation.panel_schema import Col, PanelSchema


def get_treated_ids(panel: DataFrame[PanelSchema]) -> list[int]:
    """IDs of all firms that received treatment at any point."""
    return panel[panel[Col.TRATADO_EN_T]][Col.ID_FIRMA].unique().tolist()


def get_control_ids(panel: DataFrame[PanelSchema]) -> list[int]:
    """IDs of all firms assigned as control in any cohort."""
    return panel[panel[Col.CONTROL_EN_T]][Col.ID_FIRMA].unique().tolist()


def get_nini_ids(panel: DataFrame[PanelSchema]) -> list[int]:
    """IDs of firms that were neither treated nor assigned as control."""
    nini_firms = panel.groupby(Col.ID_FIRMA).filter(
        lambda g: (~g[Col.TRATADO_EN_T].any()) & (~g[Col.CONTROL_EN_T].any())
    )
    return nini_firms[Col.ID_FIRMA].unique().tolist()


def get_treated_per_cohort(panel: DataFrame[PanelSchema]) -> dict[int, list[int]]:
    """Maps each treatment period to the IDs of firms treated in that cohort."""
    treated = panel[panel[Col.TRATADO_EN_T]]
    return treated.groupby(Col.T)[Col.ID_FIRMA].apply(list).to_dict()


def get_controls_per_cohort(panel: DataFrame[PanelSchema]) -> dict[int, list[int]]:
    """Maps each treatment period to the IDs of firms assigned as control in that cohort."""
    controls = panel[panel[Col.CONTROL_EN_T]]
    return controls.groupby(Col.T)[Col.ID_FIRMA].apply(list).to_dict()
