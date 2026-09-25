"""GateLinear model interface. The implementation will be released publicly later."""

import torch.nn as nn


class Model(nn.Module):
    """Public interface placeholder for GateLinear.

    The model implementation is not included in this release.
    Source code will be made publicly available in a future release.
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """Forecast interface; the implementation is pending public release.

        Args:
            x_enc: Historical inputs with shape [batch, lookback, channels].
            x_mark_enc: Optional historical time features.
            x_dec: Optional decoder inputs.
            x_mark_dec: Optional decoder time features.

        Returns:
            Forecasts with shape [batch, horizon, channels] once released.
        """
        # The GateLinear implementation will be released publicly later.
        raise NotImplementedError(
            "GateLinear source code is pending public release. "
            "Only the Model interface is currently available."
        )
