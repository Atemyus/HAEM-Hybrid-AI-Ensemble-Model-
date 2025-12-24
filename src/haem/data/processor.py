"""
Data Processing Module for HAEM.

This module provides utilities for:
- Grid interpolation and regridding
- Data quality control
- Missing data handling
- Field smoothing and filtering
"""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, uniform_filter

from haem.models.meteorological import AtmosphericField, ModelData, NWPModel

logger = logging.getLogger(__name__)


class GridInterpolator:
    """
    Handles grid interpolation and regridding operations.

    Supports bilinear and cubic interpolation for regridding
    model data to a common grid.
    """

    def __init__(
        self,
        target_lats: NDArray[np.float64],
        target_lons: NDArray[np.float64],
        method: str = "linear",
    ):
        """
        Initialize with target grid.

        Args:
            target_lats: Target latitude array
            target_lons: Target longitude array
            method: Interpolation method ('linear', 'nearest', 'cubic')
        """
        self.target_lats = target_lats
        self.target_lons = target_lons
        self.method = method
        self._target_mesh = np.meshgrid(target_lons, target_lats)

    def interpolate_field(
        self,
        field: AtmosphericField,
        fill_value: float = np.nan,
    ) -> NDArray[np.float64]:
        """
        Interpolate a field to the target grid.

        Args:
            field: Source atmospheric field
            fill_value: Value to use for out-of-bounds points

        Returns:
            Interpolated data on target grid
        """
        # Create interpolator from source field
        interp = RegularGridInterpolator(
            (field.lats, field.lons),
            field.data,
            method=self.method,
            bounds_error=False,
            fill_value=fill_value,
        )

        # Create target points
        target_points = np.column_stack([
            self._target_mesh[1].ravel(),  # lats
            self._target_mesh[0].ravel(),  # lons
        ])

        # Interpolate
        result = interp(target_points)
        return result.reshape(len(self.target_lats), len(self.target_lons))

    def regrid_model_data(
        self,
        model_data: ModelData,
        fields: Optional[list[str]] = None,
    ) -> ModelData:
        """
        Regrid all fields in a ModelData object to the target grid.

        Args:
            model_data: Source model data
            fields: Fields to regrid (default: all)

        Returns:
            New ModelData with regridded fields
        """
        if fields is None:
            fields = ["z500", "z850", "t850", "slp"]

        # Create new model data with same run info
        new_data = ModelData(
            run=model_data.run,
            data_quality_score=model_data.data_quality_score,
            missing_times=model_data.missing_times.copy(),
        )

        for field_name in fields:
            field_dict = getattr(model_data, field_name)
            new_dict = getattr(new_data, field_name)

            for hour, field in field_dict.items():
                interp_data = self.interpolate_field(field)

                # Create new field with interpolated data
                new_field = field.model_copy(
                    update={
                        "lats": self.target_lats,
                        "lons": self.target_lons,
                        "data": interp_data,
                    }
                )
                new_dict[hour] = new_field

        return new_data


class DataProcessor:
    """
    Comprehensive data processing utilities.

    Provides methods for:
    - Quality control
    - Smoothing and filtering
    - Anomaly detection
    - Missing data handling
    """

    @staticmethod
    def apply_gaussian_smooth(
        data: NDArray[np.float64],
        sigma: float = 1.0,
    ) -> NDArray[np.float64]:
        """
        Apply Gaussian smoothing to a 2D field.

        Args:
            data: 2D array to smooth
            sigma: Standard deviation for Gaussian kernel

        Returns:
            Smoothed array
        """
        return gaussian_filter(data, sigma=sigma, mode="nearest")

    @staticmethod
    def apply_box_filter(
        data: NDArray[np.float64],
        size: int = 3,
    ) -> NDArray[np.float64]:
        """
        Apply uniform (box) filter to a 2D field.

        Args:
            data: 2D array to filter
            size: Filter window size

        Returns:
            Filtered array
        """
        return uniform_filter(data, size=size, mode="nearest")

    @staticmethod
    def compute_gradient(
        data: NDArray[np.float64],
        dx: float,
        dy: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute spatial gradients of a 2D field.

        Args:
            data: 2D array
            dx: Grid spacing in x direction (meters or degrees)
            dy: Grid spacing in y direction

        Returns:
            Tuple of (ddata/dx, ddata/dy)
        """
        grad_y, grad_x = np.gradient(data, dy, dx)
        return grad_x, grad_y

    @staticmethod
    def compute_laplacian(
        data: NDArray[np.float64],
        dx: float,
        dy: float,
    ) -> NDArray[np.float64]:
        """
        Compute Laplacian of a 2D field.

        Args:
            data: 2D array
            dx: Grid spacing in x direction
            dy: Grid spacing in y direction

        Returns:
            Laplacian field
        """
        grad_x, grad_y = DataProcessor.compute_gradient(data, dx, dy)
        grad_xx, _ = DataProcessor.compute_gradient(grad_x, dx, dy)
        _, grad_yy = DataProcessor.compute_gradient(grad_y, dx, dy)
        return grad_xx + grad_yy

    @staticmethod
    def detect_outliers(
        data: NDArray[np.float64],
        threshold: float = 3.0,
    ) -> NDArray[np.bool_]:
        """
        Detect outliers using z-score method.

        Args:
            data: Array to check
            threshold: Number of standard deviations for outlier

        Returns:
            Boolean mask (True = outlier)
        """
        mean = np.nanmean(data)
        std = np.nanstd(data)

        if std == 0:
            return np.zeros_like(data, dtype=bool)

        z_scores = np.abs((data - mean) / std)
        return z_scores > threshold

    @staticmethod
    def fill_missing(
        data: NDArray[np.float64],
        method: str = "interpolate",
    ) -> NDArray[np.float64]:
        """
        Fill missing (NaN) values in a 2D field.

        Args:
            data: 2D array with NaN values
            method: Fill method ('interpolate', 'mean', 'nearest')

        Returns:
            Array with filled values
        """
        if not np.any(np.isnan(data)):
            return data

        result = data.copy()

        if method == "mean":
            mean_val = np.nanmean(data)
            result[np.isnan(result)] = mean_val

        elif method == "interpolate":
            from scipy.interpolate import griddata

            # Get valid and invalid points
            valid_mask = ~np.isnan(data)
            invalid_mask = np.isnan(data)

            if np.sum(valid_mask) < 4:
                # Not enough valid points, use mean
                mean_val = np.nanmean(data)
                result[invalid_mask] = mean_val
            else:
                # Create coordinate arrays
                y_coords, x_coords = np.mgrid[0:data.shape[0], 0:data.shape[1]]

                # Get valid points
                valid_points = np.column_stack([
                    x_coords[valid_mask],
                    y_coords[valid_mask]
                ])
                valid_values = data[valid_mask]

                # Get invalid points
                invalid_points = np.column_stack([
                    x_coords[invalid_mask],
                    y_coords[invalid_mask]
                ])

                # Interpolate
                filled_values = griddata(
                    valid_points,
                    valid_values,
                    invalid_points,
                    method="linear",
                )
                result[invalid_mask] = filled_values

                # Handle any remaining NaN (from extrapolation)
                still_nan = np.isnan(result)
                if np.any(still_nan):
                    result[still_nan] = np.nanmean(data)

        elif method == "nearest":
            from scipy.ndimage import distance_transform_edt

            mask = np.isnan(data)
            if np.any(mask):
                _, indices = distance_transform_edt(mask, return_indices=True)
                result = data[tuple(indices)]

        return result

    @staticmethod
    def quality_control(
        data: NDArray[np.float64],
        valid_range: tuple[float, float],
        replace_outliers: bool = True,
    ) -> tuple[NDArray[np.float64], float]:
        """
        Perform quality control on a field.

        Args:
            data: 2D array to check
            valid_range: (min, max) valid values
            replace_outliers: Whether to replace outliers with NaN

        Returns:
            Tuple of (cleaned data, quality score 0-1)
        """
        result = data.copy()

        # Check for out-of-range values
        out_of_range = (data < valid_range[0]) | (data > valid_range[1])
        n_out_of_range = np.sum(out_of_range)

        # Check for NaN/Inf
        n_invalid = np.sum(~np.isfinite(data))

        # Check for statistical outliers
        outlier_mask = DataProcessor.detect_outliers(data, threshold=4.0)
        n_outliers = np.sum(outlier_mask & ~out_of_range)

        # Calculate quality score
        total_points = data.size
        bad_points = n_out_of_range + n_invalid + n_outliers
        quality_score = 1.0 - (bad_points / total_points)

        if replace_outliers:
            result[out_of_range | ~np.isfinite(data)] = np.nan
            result = DataProcessor.fill_missing(result, method="interpolate")

        return result, quality_score


class ModelDataValidator:
    """Validates model data for physical consistency."""

    # Valid ranges for different fields
    VALID_RANGES = {
        "z500": (480, 600),  # dam
        "z850": (120, 160),  # dam
        "t850": (230, 320),  # K
        "slp": (940, 1070),   # hPa
    }

    @classmethod
    def validate(
        cls,
        model_data: ModelData,
    ) -> tuple[ModelData, dict[str, float]]:
        """
        Validate and clean model data.

        Args:
            model_data: Model data to validate

        Returns:
            Tuple of (cleaned ModelData, quality scores by field)
        """
        quality_scores: dict[str, float] = {}

        for field_name, valid_range in cls.VALID_RANGES.items():
            field_dict = getattr(model_data, field_name)
            field_scores = []

            for hour, field in field_dict.items():
                cleaned_data, score = DataProcessor.quality_control(
                    field.data,
                    valid_range=valid_range,
                    replace_outliers=True,
                )
                field.data = cleaned_data
                field_scores.append(score)

            if field_scores:
                quality_scores[field_name] = float(np.mean(field_scores))

        # Update overall quality score
        if quality_scores:
            model_data.data_quality_score = float(np.mean(list(quality_scores.values())))

        return model_data, quality_scores
