import numpy as np

from deepness.common.processing_parameters.standardization_parameters import StandardizationParameters


def limit_channels_number(tiles_batched: np.array, limit: int) -> np.array:
    """ Limit the number of channels in the input image to the model

    :param tiles_batched: Batch of tiles
    :param limit: Number of channels to keep
    :return: Batch of tiles with limited number of channels
    """
    return tiles_batched[:, :, :, :limit]

def normalize_band(band):
    """Normalize a single band with contrast stretching, handling bad values."""
    valid_mask = np.isfinite(band) & (band > -1e10)
    if not np.any(valid_mask):
        return np.zeros_like(band, dtype=np.uint8)

    p2, p98 = np.percentile(band[valid_mask], (2, 98))
    norm = np.clip((band - p2) / (p98 - p2), 0, 1)
    return (norm * 255).astype(np.float32)

def normalize_6_channels(tiles_batched): 
    # tiles_batched shape: (4,256,256,6)
    norm_tiles = []
    for tile_index in range(tiles_batched.shape[0]):
        tile = tiles_batched[tile_index]  # shape: (256,256,6)
        norm_tile = [normalize_band(tile[:,:,i]) for i in range(tile.shape[2])]
        norm_tile = np.stack(norm_tile, axis=0) / 255 # shape: (6,256,256)
        norm_tiles.append(norm_tile)
    return np.stack(norm_tiles, axis=0) # shape: (4,6,256,256) NCHW

def normalize_values_to_01(tiles_batched: np.array) -> np.array:
    """ Normalize the values of the input image to the model to the range [0, 1]

    :param tiles_batched: Batch of tiles
    :return: Batch of tiles with values in the range [0, 1], in float32
    """
    return np.float32(tiles_batched * 1./255.)

def standardize_values(tiles_batched: np.array, params: StandardizationParameters) -> np.array:
    """ Standardize the input image to the model

    :param tiles_batched: Batch of tiles
    :param params: Parameters for standardization of type STANDARIZE_PARAMS
    :return: Batch of tiles with standardized values
    """
    print(f'params.mean: {params.mean}')
    print(f'params.std: {params.std}')

    return (tiles_batched - params.mean) / params.std


def transpose_nhwc_to_nchw(tiles_batched: np.array) -> np.array:
    """ Transpose the input image from NHWC to NCHW

    :param tiles_batched: Batch of tiles in NHWC format
    :return: Batch of tiles in NCHW format
    """
    return np.transpose(tiles_batched, (0, 3, 1, 2))
