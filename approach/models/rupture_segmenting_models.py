from functools import partial
from typing import Type, Dict, Any

from sklearn.base import BaseEstimator

from ..files import DenoiseParameters, RuptureParams, RuptureSegmentAlgorithms
from ..models import (
    SegmentingModel,
    KBinSegSegmentModel,
    LinearRegressionValueModel,
    LinearRegressionTimeModel,
)
from . import (
    KBottomUpSegmentModel,
    KPeltSegmentModel,
    KDynpSegmentModel,
    KWindowSegmentModel,
    KKernelCPDSegmentModel,
    ChangePointSegmentModel,
)


class RuptureSegmentingModel(SegmentingModel):
    name = "RuptureSegmentingModel"
    num_segments: int

    segment_algorithm_to_segment_model: Dict[
        RuptureSegmentAlgorithms, Type[ChangePointSegmentModel]
    ] = {
        RuptureSegmentAlgorithms.binseg: KBinSegSegmentModel,
        RuptureSegmentAlgorithms.bottom_up: KBottomUpSegmentModel,
        RuptureSegmentAlgorithms.dynp: KDynpSegmentModel,
        RuptureSegmentAlgorithms.kernel_cpd: KKernelCPDSegmentModel,
        RuptureSegmentAlgorithms.pelt: KPeltSegmentModel,
        RuptureSegmentAlgorithms.window: KWindowSegmentModel,
    }

    def __init__(
        self,
        rupture_params: RuptureParams,
        reg_model_type: Type[BaseEstimator],
        algorithm: RuptureSegmentAlgorithms,
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
            self.segment_algorithm_to_segment_model[algorithm](
                rupture_params, use_preprocessed=use_preprocessed
            ),
            *args,
            **kwargs,
        )
        self.num_segments = (
            rupture_params.num_segments
        )  # todo kinda unnecessary to save this..?
        self.use_preprocessed = use_preprocessed


class KBottomUpSegmentingModel(RuptureSegmentingModel):
    name = "KBottomUpSegmentingModel"
    num_segments: int

    def __init__(
        self,
        rupture_params: RuptureParams,
        reg_model_type: Type[BaseEstimator],
        use_preprocessed: bool | DenoiseParameters = False,
        reg_model_params: Dict[str, Any] = None,
        *args,
        **kwargs
    ):
        super().__init__(
            rupture_params,
            reg_model_type,
            RuptureSegmentAlgorithms.bottom_up,
            use_preprocessed,
            reg_model_params,
        )


class KBinSegSegmentingModel(RuptureSegmentingModel):
    name = "KBinSegSegmentingModel"
    num_segments: int

    def __init__(
        self,
        rupture_params: RuptureParams,
        reg_model_type: Type[BaseEstimator],
        use_preprocessed: bool | DenoiseParameters = False,
        reg_model_params: Dict[str, Any] = None,
        *args,
        **kwargs
    ):
        super().__init__(
            rupture_params,
            reg_model_type,
            RuptureSegmentAlgorithms.binseg,
            use_preprocessed,
            reg_model_params,
        )


class KPeltSegmentingModel(RuptureSegmentingModel):
    name = "KPeltSegmentingModel"
    num_segments: int

    def __init__(
        self,
        rupture_params: RuptureParams,
        reg_model_type: Type[BaseEstimator],
        use_preprocessed: bool | DenoiseParameters = False,
        reg_model_params: Dict[str, Any] = None,
        *args,
        **kwargs
    ):
        super().__init__(
            rupture_params,
            reg_model_type,
            RuptureSegmentAlgorithms.pelt,
            use_preprocessed,
            reg_model_params,
        )


class KDynpSegmentingModel(RuptureSegmentingModel):
    name = "KDynpSegmentingModel"
    num_segments: int

    def __init__(
        self,
        rupture_params: RuptureParams,
        reg_model_type: Type[BaseEstimator],
        use_preprocessed: bool | DenoiseParameters = False,
        reg_model_params: Dict[str, Any] = None,
        *args,
        **kwargs
    ):
        super().__init__(
            rupture_params,
            reg_model_type,
            RuptureSegmentAlgorithms.dynp,
            use_preprocessed,
            reg_model_params,
        )


class KWindowSegmentingModel(RuptureSegmentingModel):
    name = "KWindowSegmentingModel"
    num_segments: int

    def __init__(
        self,
        rupture_params: RuptureParams,
        reg_model_type: Type[BaseEstimator],
        use_preprocessed: bool | DenoiseParameters = False,
        reg_model_params: Dict[str, Any] = None,
        *args,
        **kwargs
    ):
        super().__init__(
            rupture_params,
            reg_model_type,
            RuptureSegmentAlgorithms.window,
            use_preprocessed,
            reg_model_params,
        )


class KKernelCPDSegmentingModel(RuptureSegmentingModel):
    name = "KKernelCPDSegmentingModel"
    num_segments: int

    def __init__(
        self,
        rupture_params: RuptureParams,
        reg_model_type: Type[BaseEstimator],
        use_preprocessed: bool | DenoiseParameters = False,
        reg_model_params: Dict[str, Any] = None,
        *args,
        **kwargs
    ):
        super().__init__(
            rupture_params,
            reg_model_type,
            RuptureSegmentAlgorithms.kernel_cpd,
            use_preprocessed,
            reg_model_params,
        )
