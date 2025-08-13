from dataclasses import dataclass
from typing import Optional

from deepness.common.processing_parameters.map_processing_parameters import MapProcessingParameters


@dataclass
class ClassifyChipParameters(MapProcessingParameters):
    """
    Parameters for Classifying Chips obtained from UI.
    """

    raster_id: Optional[str]  # id for map layer
    vector_id: Optional[str]  # id for bounding box layer
    config: dict # model configuration 
    # config keys: 'model_name', 'weights_ckpt', 'class_names', 'normalization_mean',
    # 'normalization_std', 'model_arch', 'n_channels', 'image_size'