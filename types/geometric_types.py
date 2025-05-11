import numpy as np
from typing import Any, Sequence, Tuple, Union
from numpy.typing import NDArray


class TypedNDArray(np.ndarray):
    """
    Base class for fixed-shape, strongly-typed numpy arrays.
    Subclasses must define `expected_shape` and `expected_ndim`.
    """
    expected_ndim: int
    expected_shape: Tuple[int, ...]

    def __new__(
        cls,
        input_array: Union[NDArray[Any], Sequence[Any]],
        /,
        dtype: Any = float,
        copy: bool = False,
    ):
        # Create a base array, enforce dtype
        arr = np.array(input_array, dtype=dtype, copy=copy)
        # Validate shape
        if arr.ndim != cls.expected_ndim or arr.shape != cls.expected_shape:
            raise ValueError(
                f"{cls.__name__} must have shape {cls.expected_shape}, got {arr.shape}"
            )
        # View as subclass
        obj = arr.view(cls)
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        # Called after new view/copy; shape is validated in __new__
        if obj is None:
            return

    @classmethod
    def _wrap_class(cls, arr: np.ndarray) -> Union[np.ndarray, "TypedNDArray"]:
        """
        If arr matches the expected shape, view it as cls, else return as base ndarray.
        """
        if arr.ndim == cls.expected_ndim and arr.shape == cls.expected_shape:
            return arr.view(cls)
        return arr

    def to_numpy(self) -> np.ndarray:
        """
        Convert to a plain numpy.ndarray (copy).
        """
        return np.array(self)


class Matrix3x3(TypedNDArray):
    expected_ndim = 2
    expected_shape = (3, 3)

    def flatten(self, order: str = 'C') -> np.ndarray:
        return np.ndarray.flatten(self, order)


class RotationMatrix(Matrix3x3):
    """
    3x3 rotation matrix (orthonormal). Does not enforce orthonormality by default.
    """
    pass


class Rotation6D(TypedNDArray):
    expected_ndim = 1
    expected_shape = (6,)


class Quaternion(TypedNDArray):
    expected_ndim = 1
    expected_shape = (4,)


class Pose4x4(TypedNDArray):
    expected_ndim = 2
    expected_shape = (4, 4)

    @property
    def rotation_matrix(self) -> Matrix3x3:
        return Matrix3x3(self[:3, :3])

    @property
    def translation_vector(self) -> np.ndarray:
        return np.array(self[:3, 3])

    def flatten(self, order: str = 'C') -> np.ndarray:
        return np.ndarray.flatten(self, order)
