from . import ChangePointSegmentModel
from ..files import RuptureSegmentAlgorithms, DenoiseParameters, RuptureParams


class KBinSegSegmentModel(ChangePointSegmentModel):
    name = "KBinSegSegmentModel"

    def __init__(
        self,
        rupture_params: RuptureParams,
        use_preprocessed: bool | DenoiseParameters = False,
    ):
        super().__init__(
            RuptureSegmentAlgorithms.binseg,
            rupture_params,
            use_preprocessed=use_preprocessed,
        )


class KPeltSegmentModel(ChangePointSegmentModel):
    name = "KPeltSegmentModel"

    def __init__(
        self,
        rupture_params: RuptureParams,
        use_preprocessed: bool | DenoiseParameters = False,
    ):
        super().__init__(
            RuptureSegmentAlgorithms.pelt,
            rupture_params,
            use_preprocessed=use_preprocessed,
        )


class KWindowSegmentModel(ChangePointSegmentModel):
    name = "KWindowSegmentModel"

    def __init__(
        self,
        rupture_params: RuptureParams,
        use_preprocessed: bool | DenoiseParameters = False,
    ):
        super().__init__(
            RuptureSegmentAlgorithms.window,
            rupture_params,
            use_preprocessed=use_preprocessed,
        )


class KKernelCPDSegmentModel(ChangePointSegmentModel):
    name = "KKernelCPDSegmentModel"

    def __init__(
        self,
        rupture_params: RuptureParams,
        use_preprocessed: bool | DenoiseParameters = False,
    ):
        super().__init__(
            RuptureSegmentAlgorithms.kernel_cpd,
            rupture_params,
            use_preprocessed=use_preprocessed,
        )


class KDynpSegmentModel(ChangePointSegmentModel):
    name = "KDynpSegmentModel"

    def __init__(
        self,
        rupture_params: RuptureParams,
        use_preprocessed: bool | DenoiseParameters = False,
    ):
        super().__init__(
            RuptureSegmentAlgorithms.dynp,
            rupture_params,
            use_preprocessed=use_preprocessed,
        )


class KBottomUpSegmentModel(ChangePointSegmentModel):
    name = "KBottomUpSegmentModel"

    def __init__(
        self,
        rupture_params: RuptureParams,
        use_preprocessed: bool | DenoiseParameters = False,
    ):
        super().__init__(
            RuptureSegmentAlgorithms.bottom_up,
            rupture_params,
            use_preprocessed=use_preprocessed,
        )
