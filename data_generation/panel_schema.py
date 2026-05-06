import pandas as pd
import pandera.pandas as pa

from pandera.typing import Series

class PanelSchema(pa.DataFrameModel):
    id_firma:     Series[int]
    inicio_firma: Series[int]
    t:            Series[int]
    tratado_en_t: Series[bool]
    control_en_t: Series[bool]

    @pa.dataframe_check
    def at_most_one_treatment_per_firm(cls, df: pd.DataFrame) -> bool:
        # Las firmas tratadas solo pueden ser tratadas una vez
        return (df.groupby(Col.ID_FIRMA)[Col.TRATADO_EN_T].sum() <= 1).all()

    @pa.dataframe_check
    def treated_firms_never_control(cls, df: pd.DataFrame) -> bool:
        # Las firmas tratadas no pueden ser control en ningún período
        treated_ids = df.loc[df[Col.TRATADO_EN_T], Col.ID_FIRMA].unique()
        return not df[df[Col.ID_FIRMA].isin(treated_ids)][Col.CONTROL_EN_T].any()


class Col:
    ID_FIRMA     = "id_firma"
    INICIO_FIRMA = "inicio_firma"
    T            = "t"
    TRATADO_EN_T = "tratado_en_t"
    CONTROL_EN_T = "control_en_t"
