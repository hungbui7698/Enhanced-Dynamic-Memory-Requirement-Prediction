from functools import partial
from typing import Type, Any, Dict

from sklearn.base import BaseEstimator

from ..files import DenoiseParameters
from ..models import (
    SegmentingModel,
    LinearRegressionTimeModel,
    LinearRegressionValueModel,
    KEqualSegmentModel,
)


class KEqualSegmentingModel(SegmentingModel):
    name = "KEqualSegmentingModel"
    num_segments: int

    def __init__(
        self,
        k: int,
        reg_model_type: Type[BaseEstimator],
        use_preprocessed: bool | DenoiseParameters = False,
        reg_model_params: Dict[str, Any] = None,
        *args,
        **kwargs
    ):
        if reg_model_params is None:
            reg_model_params = {}
        super().__init__(
            LinearRegressionTimeModel(
                reg_model_type=reg_model_type,
                use_preprocessed=use_preprocessed,
                reg_model_params=reg_model_params,
            ),
            partial(
                LinearRegressionValueModel,
                reg_model_type=reg_model_type,
                reg_model_params=reg_model_params,
            ),
            KEqualSegmentModel(k, use_preprocessed=use_preprocessed),
            *args,
            **kwargs,
        )
        self.num_segments = k  # todo kinda unnecessary to save this..?
        self.use_preprocessed = use_preprocessed
