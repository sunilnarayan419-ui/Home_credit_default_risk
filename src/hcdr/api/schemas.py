"""Pydantic request/response schemas.

``ApplicationFeatures`` mirrors the 32 raw application columns the models are trained on (everything in
``train.csv`` except ``SK_ID_CURR`` and ``TARGET``); ``tests/test_api_contract.py`` checks it against the
feature-engineering code. Field names, units and meaning follow the competition data. Fields that are
legitimately missing in real applications (see ``experiments/EXPERIMENTS.md``) are optional and are scored as
missing values, which the tree models handle natively. ``DAYS_*`` follow the dataset's convention: days relative
to the application date, negative = in the past. ``DAYS_EMPLOYED`` may hold the dataset's 365243 sentinel.
Unknown fields are rejected so typos cannot be silently ignored.

``typing.Optional`` is used (not ``X | None``) because the package supports Python 3.9.
"""
from __future__ import annotations

from typing import Annotated, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

Text = Annotated[str, Field(min_length=1, max_length=100)]
Amount = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Number = Annotated[float, Field(allow_inf_nan=False)]
Count = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Flag = Annotated[int, Field(ge=0, le=1)]


class ApplicationFeatures(BaseModel):
    """One loan application, with the raw columns of the competition's ``train.csv``/``test.csv``."""

    model_config = ConfigDict(extra="forbid")

    NAME_CONTRACT_TYPE: Text
    CODE_GENDER: Text
    NAME_INCOME_TYPE: Text
    NAME_EDUCATION_TYPE: Text
    NAME_FAMILY_STATUS: Text
    NAME_HOUSING_TYPE: Text
    OCCUPATION_TYPE: Optional[Text] = None
    ORGANIZATION_TYPE: Text
    CNT_CHILDREN: Annotated[int, Field(ge=0)]
    AMT_INCOME_TOTAL: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    AMT_CREDIT: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    AMT_ANNUITY: Optional[Amount] = None
    AMT_GOODS_PRICE: Optional[Amount] = None
    REGION_POPULATION_RELATIVE: Amount
    DAYS_BIRTH: Annotated[float, Field(lt=0, allow_inf_nan=False)]
    DAYS_EMPLOYED: Number
    DAYS_REGISTRATION: Number
    DAYS_ID_PUBLISH: Number
    OWN_CAR_AGE: Optional[Amount] = None
    FLAG_MOBIL: Flag
    FLAG_EMP_PHONE: Flag
    FLAG_WORK_PHONE: Flag
    CNT_FAM_MEMBERS: Optional[Count] = None
    REGION_RATING_CLIENT: Annotated[int, Field(ge=1, le=3)]
    EXT_SOURCE_1: Optional[Score] = None
    EXT_SOURCE_2: Optional[Score] = None
    EXT_SOURCE_3: Optional[Score] = None
    DAYS_LAST_PHONE_CHANGE: Optional[Number] = None
    AMT_REQ_CREDIT_BUREAU_HOUR: Optional[Count] = None
    AMT_REQ_CREDIT_BUREAU_MON: Optional[Count] = None
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[Count] = None
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[Count] = None


RAW_COLUMNS = tuple(ApplicationFeatures.model_fields)
MAX_BATCH_SIZE = 1000


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: List[ApplicationFeatures] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class PredictionResponse(BaseModel):
    default_probability: float = Field(
        description="Blend score for TARGET = 1 (payment difficulties), in [0, 1]. "
        "A ranking score validated by ROC-AUC; it is not shown to be a calibrated probability."
    )


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse] = Field(description="Same order as the request records.")


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when a model is loaded, otherwise 'degraded'.")
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    hcdr_version: str
    models: List[str]
    blend: str
    n_train_rows: int
    n_raw_columns: int
    n_features: int
    feature_groups: List[str]
    trained_at: str
    library_versions: Dict[str, str]


class ErrorResponse(BaseModel):
    detail: str
