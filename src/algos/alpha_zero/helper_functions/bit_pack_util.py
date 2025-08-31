import numpy as np

# ========= Bit-pack utilities =========
def _pack_bits_bool2d(arr2d: np.ndarray) -> np.ndarray:
    """
    Pack a 2D 0/1 or bool array into a 1D uint8 bit vector using bitorder='big'.
    """
    # Ensure uint8 0/1
    a = arr2d.astype(np.uint8, copy=False)
    return np.packbits(a.reshape(-1), bitorder='big')

def _unpack_bits_to_2d(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    Unpack a 1D uint8 bit vector to a 2D uint8 array (0/1) with given shape.
    """
    flat = np.unpackbits(bits, bitorder='big')
    need = rows * cols
    if flat.size > need:
        flat = flat[:need]
    return flat.reshape((rows, cols)).astype(np.uint8, copy=False)

def _bit_clear_inplace(bits: np.ndarray, idx: int) -> None:
    """
    Clear (set to 0) the bit at flat index idx in the packed array (bitorder='big').
    Uses a non-negative mask to avoid OverflowError from bitwise NOT on Python ints.
    """
    byte_i = idx // 8
    off    = idx % 8
    # Build a clear mask: 0xFF with target bit cleared
    clear_mask = np.uint8(0xFF ^ (1 << (7 - off)))
    bits[byte_i] &= clear_mask

def _bit_set_inplace(bits: np.ndarray, idx: int) -> None:
    """
    Set (to 1) the bit at flat index idx in the packed array (bitorder='big').
    """
    byte_i = idx // 8
    off    = idx % 8
    bits[byte_i] |= np.uint8(1 << (7 - off))