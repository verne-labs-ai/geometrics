import numpy as np
from typing import Any, Sequence, Tuple, Union
from numpy.typing import NDArray


class TypedNDArray(np.ndarray):
    """
    Base class for fixed-shape, strongly-typed numpy arrays.

    Subclasses must define:
        expected_ndim (int): Expected number of dimensions.
        expected_shape (Tuple[int, ...]): Expected shape.

    Provides validation on creation and utility methods for conversion.
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
        """
        Create a new instance of the subclass from input data.

        Parameters:
            input_array (np.ndarray or Sequence[Any]): Input data to instantiate the array.
            dtype (Any, optional): Data type for the array. Defaults to float.
            copy (bool, optional): Whether to copy the data. Defaults to False.

        Returns:
            TypedNDArray: An instance of the subclass with validated dimensions and shape.

        Raises:
            ValueError: If the input array does not match expected dimensions or shape.
        """
        arr = np.array(input_array, dtype=dtype, copy=copy)
        if arr.ndim != cls.expected_ndim or arr.shape != cls.expected_shape:
            raise ValueError(
                f"{cls.__name__} must have shape {cls.expected_shape}, got {arr.shape}"
            )
        return arr.view(cls)

    def __array_finalize__(self, obj: Any) -> None:
        """
        Finalize the array view. No extra validation needed; shape was checked in __new__.

        Parameters:
            obj (Any): The base object of the array view or None.
        """
        if obj is None:
            return

    @classmethod
    def _wrap_class(cls, arr: np.ndarray) -> Union[np.ndarray, "TypedNDArray"]:
        """
        If the input array matches expected shape, view it as this class.

        Parameters:
            arr (np.ndarray): Input numpy array.

        Returns:
            TypedNDArray or np.ndarray: Wrapped instance or original array if shape mismatches.
        """
        if arr.ndim == cls.expected_ndim and arr.shape == cls.expected_shape:
            return arr.view(cls)
        return arr

    def to_numpy(self) -> np.ndarray:
        """
        Convert to a plain numpy.ndarray.

        Returns:
            np.ndarray: Copy of this array as a base numpy.ndarray.
        """
        return np.array(self)


class Matrix3x3(TypedNDArray):
    """
    Fixed-size 3x3 numpy array representing a matrix.
    """
    expected_ndim = 2
    expected_shape = (3, 3)

    def flatten(self, order: str = 'C') -> np.ndarray:
        """
        Flatten the 3x3 matrix into a 1D array of length 9.

        Parameters:
            order (str, optional): Memory layout order, 'C' or 'F'. Defaults to 'C'.

        Returns:
            np.ndarray: Flattened array of shape (9,).
        """
        return np.ndarray.flatten(self, order)


class RotationMatrix(Matrix3x3):
    """
    3x3 rotation matrix (orthonormal).
    Does not enforce orthonormality by default.
    """
    pass


class Rotation6D(TypedNDArray):
    """
    6D rotation representation array of shape (6,).
    Often used as continuous rotation encoding.
    """
    expected_ndim = 1
    expected_shape = (6,)


class Quaternion(TypedNDArray):
    """
    Quaternion representing rotation, array of shape (4,).
    Format: (w, x, y, z) or (x, y, z, w) depending on convention.
    """
    expected_ndim = 1
    expected_shape = (4,)


class Pose4x4(TypedNDArray):
    """
    4x4 homogeneous transformation matrix.

    Provides properties to extract rotation and translation components.
    """
    expected_ndim = 2
    expected_shape = (4, 4)

    @property
    def rotation_matrix(self) -> Matrix3x3:
        """
        Extract the upper-left 3x3 rotation component.

        Returns:
            Matrix3x3: Rotation matrix part of the pose.
        """
        return Matrix3x3(self[:3, :3])

    @property
    def translation_vector(self) -> np.ndarray:
        """
        Extract the translation component.

        Returns:
            np.ndarray: Translation vector of shape (3,).
        """
        return np.array(self[:3, 3])

    def flatten(self, order: str = 'C') -> np.ndarray:
        """
        Flatten the 4x4 pose matrix into a 1D array of length 16.

        Parameters:
            order (str, optional): Memory layout order, 'C' or 'F'. Defaults to 'C'.

        Returns:
            np.ndarray: Flattened array of shape (16,).
        """
        return np.ndarray.flatten(self, order)
