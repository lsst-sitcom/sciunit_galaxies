import operator
from typing import cast

from lsst.analysis.tools.actions.vector import SelectorBase
from lsst.analysis.tools.interfaces import KeyedData, KeyedDataSchema, Vector
from lsst.pex.config import Field

bands_color = {"i": "z", "r": "i", "g": "r"}


class DecalsSelector(SelectorBase):
    """Return a mask corresponding to an applied threshold."""

    op = Field[str](doc="Operator name.")
    maskbits = Field[str](doc="masked bits column name", default="refcat_maskbits")
    threshold = Field[str](doc="Threshold to apply.", default="PSF")
    vectorKey = Field[str](doc="Name of column", default="refcat_type")

    def getInputSchema(self) -> KeyedDataSchema:
        return ((self.vectorKey, Vector), (self.maskbits, Vector))

    def __call__(self, data: KeyedData, **kwargs) -> Vector:
        mask = data[self.maskbits] == 0
        if self.op:
            mask &= getattr(operator, self.op)(data[self.vectorKey], self.threshold)
        return cast(Vector, mask)
