import numpy as np

# Try CuPy first, fall back to NumPy with Numba if unavailable
try:
    import cupy as cp
    HAS_CUPY = True
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except ImportError:
    HAS_CUPY = False
    GPU_AVAILABLE = False
    cp = None

# Try Numba JIT for CPU fallback
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    jit = None
    prange = range


class Grid:
    """
    Abstraction over GPU/CPU array with automatic backend selection.
    Provides unified interface for grid operations regardless of backend.
    """
    __slots__ = ('_data', '_backend', '_shape', '_dtype')

    def __init__(self, data, backend='auto'):
        """
        Args:
            data: numpy.ndarray or cupy.ndarray
            backend: 'auto', 'numpy', 'cupy'
        """
        self._data = data
        self._shape = data.shape
        self._dtype = data.dtype

        if backend == 'auto':
            if HAS_CUPY and isinstance(data, cp.ndarray):
                self._backend = 'cupy'
            else:
                self._backend = 'numpy'
        else:
            self._backend = backend

    @property
    def data(self):
        """Get underlying array (cupy.ndarray or numpy.ndarray)"""
        return self._data

    @property
    def shape(self):
        return self._shape

    @property
    def backend(self):
        return self._backend

    def to_cpu(self):
        """Convert to NumPy array on CPU"""
        if self._backend == 'cupy':
            return cp.asnumpy(self._data)
        return self._data.copy()

    def to_gpu(self):
        """Convert to CuPy array on GPU"""
        if self._backend == 'cupy':
            return self._data
        if HAS_CUPY:
            return cp.asarray(self._data)
        raise RuntimeError("CuPy not available")

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def zeros_like(self):
        return Grid(np.zeros(self._shape, dtype=self._dtype), backend=self._backend)

    def copy(self):
        return Grid(self._data.copy(), backend=self._backend)


def get_array_module(backend):
    """Return the appropriate array module (numpy or cupy)"""
    if backend == 'cupy' and HAS_CUPY:
        return cp
    return np


def ensure_gpu(arr):
    """Ensure array is on GPU (CuPy) if available, else return as-is"""
    if HAS_CUPY and not isinstance(arr, cp.ndarray):
        return cp.asarray(arr)
    return arr


def ensure_cpu(arr):
    """Ensure array is on CPU (NumPy)"""
    if HAS_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr
