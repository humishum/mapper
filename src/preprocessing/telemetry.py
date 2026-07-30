"""GoPro telemetry extraction using gopro-py library."""

from pathlib import Path
from typing import Optional, Tuple
import logging
import numpy as np

from ..core.types import GPSTrack, IMUData

logger = logging.getLogger(__name__)


class TelemetryExtractor:
    """
    Extract telemetry data from GoPro videos using the gopro-py library.

    Extracts:
    - GPS track (latitude, longitude, altitude, timestamps)
    - IMU data (accelerometer, gyroscope, timestamps)
    - Camera orientations (quaternions, Hero8+)

    The gopro-py library is expected to be installed from:
    ~/repos/gopro-py/
    """

    def __init__(
        self,
        *,
        min_gps_fix: int = 3,
        max_gps_accuracy_m: float = 20.0,
        max_gps_position_dop: float = 5.0,
    ):
        """Initialize telemetry extractor with publication-quality GPS gates."""
        self._gopropy_available = None
        self.min_gps_fix = int(min_gps_fix)
        self.max_gps_accuracy_m = float(max_gps_accuracy_m)
        self.max_gps_position_dop = float(max_gps_position_dop)
        self._check_gopropy()

    def _check_gopropy(self) -> bool:
        """Check if gopro-py is available."""
        if self._gopropy_available is not None:
            return self._gopropy_available

        try:
            import gopropy  # noqa: F401

            self._gopropy_available = True
            logger.debug("gopro-py library available")
        except ImportError:
            self._gopropy_available = False
            logger.warning(
                "gopro-py library not found. "
                "Install from ~/repos/gopro-py/ for telemetry extraction."
            )

        return self._gopropy_available

    def extract_gps_imu(
        self, video_path: Path
    ) -> Tuple[Optional[GPSTrack], Optional[IMUData]]:
        """Extract GPS and IMU data from a GoPro video."""
        video_path = Path(video_path)

        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return None, None

        if not self._gopropy_available:
            logger.warning("Skipping telemetry extraction - gopro-py not available")
            return None, None

        try:
            import gopropy

            logger.info(f"Extracting telemetry from {video_path.name}")
            telemetry = gopropy.load(str(video_path))
            logger.debug("Telemetry streams: %s", telemetry.list_streams())
            gps_track = self._extract_gps(telemetry)
            imu_data = self._extract_imu(telemetry)

            return gps_track, imu_data

        except Exception as e:
            logger.warning(f"Failed to extract telemetry: {e}")
            return None, None

    # Compatibility alias used by the opt-in integration test and older tools.
    extract = extract_gps_imu

    def _extract_gps(self, telemetry) -> Optional[GPSTrack]:
        """Extract GPS track from telemetry."""
        try:
            # Import StreamNotFoundError here since it's only available when gopro-py is installed
            try:
                from gopropy import StreamNotFoundError
            except ImportError:
                # If not available, use a generic exception
                StreamNotFoundError = Exception

            # Try different GPS stream names
            gps_stream = None
            for name in [
                "GPS",
                "GPS5",
                "GPS9",
                "GPS (Lat., Long., Alt., ...)",
                "GPS (Lat., Long., Alt., 2D speed, 3D speed)",
            ]:
                try:
                    gps_stream = telemetry.get_stream(name)
                    logger.debug("Found GPS stream: %s", name)
                    break
                except (KeyError, AttributeError, StreamNotFoundError):
                    continue

            if gps_stream is None:
                logger.debug("No GPS stream found in telemetry")
                return None

            # Recent gopropy versions expose GPSF/GPSP and stream validity via
            # include_quality.  Fall back cleanly for older local checkouts.
            try:
                df = gps_stream.to_dataframe(include_quality=True)
            except TypeError:
                df = gps_stream.to_dataframe()

            # Find latitude/longitude columns
            lat_col = self._find_column(df.columns, "lat", "latitude")
            lon_col = self._find_column(df.columns, "lon", "longitude")
            alt_col = self._find_column(df.columns, "alt", "altitude")

            if lat_col is None or lon_col is None:
                logger.warning("Could not find lat/lon columns in GPS data")
                return None

            # Extract data
            latitudes = df[lat_col].values.astype(np.float64)
            longitudes = df[lon_col].values.astype(np.float64)

            altitudes = None
            if alt_col is not None:
                altitudes = df[alt_col].values.astype(np.float64)

            timestamps = None
            if "timestamp" in df.columns:
                timestamps = df["timestamp"].values.astype(np.float64)

            fix_col = self._find_column(df.columns, "fix", "gps_fix")
            fixes = (
                df[fix_col].values.astype(np.float64) if fix_col is not None else None
            )

            fourcc = str(getattr(gps_stream, "metadata", {}).get("fourcc", ""))
            dop_col = self._find_column(df.columns, "dop", "pdop", "position_dop")
            position_dops = (
                df[dop_col].values.astype(np.float64) if dop_col is not None else None
            )
            # GPS9 embeds DOP in its eighth field.  GPS5's separate GPSP value
            # is exposed by gopropy as a positional precision in metres.
            accuracy_col = None
            if fourcc != "GPS9":
                accuracy_col = self._find_column(
                    df.columns,
                    "gps_error_m",
                    "horizontal_accuracy_m",
                    "accuracy",
                    "precision",
                )
            accuracies = (
                df[accuracy_col].values.astype(np.float64)
                if accuracy_col is not None
                else None
            )

            valid_mask = (
                np.isfinite(latitudes)
                & np.isfinite(longitudes)
                & (latitudes >= -90.0)
                & (latitudes <= 90.0)
                & (longitudes >= -180.0)
                & (longitudes <= 180.0)
                & ~((latitudes == 0.0) & (longitudes == 0.0))
            )
            if altitudes is not None:
                valid_mask &= np.isfinite(altitudes)
            if timestamps is not None:
                valid_mask &= np.isfinite(timestamps)
            valid_col = self._find_column(df.columns, "valid")
            if valid_col is not None:
                valid_mask &= df[valid_col].values.astype(bool)
            if fixes is not None:
                valid_mask &= np.isfinite(fixes) & (fixes >= self.min_gps_fix)
            if accuracies is not None:
                valid_mask &= (
                    np.isfinite(accuracies)
                    & (accuracies >= 0.0)
                    & (accuracies <= self.max_gps_accuracy_m)
                )
            if position_dops is not None:
                valid_mask &= (
                    np.isfinite(position_dops)
                    & (position_dops >= 0.0)
                    & (position_dops <= self.max_gps_position_dop)
                )
            if not valid_mask.any():
                logger.warning("No GPS readings passed coordinate/fix/precision gates")
                return None

            latitudes = latitudes[valid_mask]
            longitudes = longitudes[valid_mask]
            if altitudes is not None:
                altitudes = altitudes[valid_mask]
            if timestamps is not None:
                timestamps = timestamps[valid_mask]
            if accuracies is not None:
                accuracies = accuracies[valid_mask]
            if fixes is not None:
                fixes = fixes[valid_mask]
            if position_dops is not None:
                position_dops = position_dops[valid_mask]

            logger.info(
                "Extracted %d quality GPS samples (%d rejected)",
                len(latitudes),
                int(np.count_nonzero(~valid_mask)),
            )

            return GPSTrack(
                latitudes=latitudes,
                longitudes=longitudes,
                altitudes=altitudes,
                timestamps=timestamps,
                accuracies=accuracies,
                fixes=fixes,
                position_dops=position_dops,
            )

        except Exception as e:
            logger.warning(f"Failed to extract GPS data: {e}")
            return None

    @staticmethod
    def _find_column(columns, *candidates):
        """Find a telemetry column by exact normalized name, then suffix."""

        normalized = {
            str(column).lower().strip().replace(" ", "_"): column for column in columns
        }
        for candidate in candidates:
            key = candidate.lower().strip().replace(" ", "_")
            if key in normalized:
                return normalized[key]
        for candidate in candidates:
            key = candidate.lower().strip().replace(" ", "_")
            for normalized_name, original in normalized.items():
                if normalized_name.endswith(f"_{key}"):
                    return original
        return None

    def _extract_imu(self, telemetry) -> Optional[IMUData]:
        """Extract IMU data from telemetry."""
        try:
            # Get accelerometer
            accel_stream = None
            for name in ["Accelerometer", "ACCL"]:
                try:
                    accel_stream = telemetry.get_stream(name)
                    break
                except (KeyError, AttributeError):
                    continue

            if accel_stream is None:
                logger.debug("No accelerometer stream found")
                return None

            # Get gyroscope
            gyro_stream = None
            for name in ["Gyroscope", "GYRO"]:
                try:
                    gyro_stream = telemetry.get_stream(name)
                    break
                except (KeyError, AttributeError):
                    continue

            if gyro_stream is None:
                logger.debug("No gyroscope stream found")
                return None

            # Extract accelerometer data
            accel_data, accel_timestamps = self._stream_to_xyz(accel_stream)

            # Extract gyroscope data
            gyro_data, gyro_timestamps = self._stream_to_xyz(gyro_stream)

            # Resample to common timestamps if needed
            # For now, use accelerometer timestamps as reference
            if len(accel_timestamps) != len(gyro_timestamps):
                # Interpolate gyro to accel timestamps
                gyro_data = self._interpolate_to_timestamps(
                    gyro_data, gyro_timestamps, accel_timestamps
                )

            timestamps = accel_timestamps

            # Try to get gravity vectors (Hero8+)
            gravity_vectors = None
            try:
                grav_stream = telemetry.get_stream("Gravity Vector")
                gravity_vectors = grav_stream.data
                # Interpolate if needed
                if len(grav_stream.timestamps) != len(timestamps):
                    gravity_vectors = self._interpolate_to_timestamps(
                        gravity_vectors, grav_stream.timestamps, timestamps
                    )
            except (KeyError, AttributeError):
                pass

            # Try to get camera orientations (Hero8+)
            orientations = None
            try:
                cori_stream = telemetry.get_stream("CameraOrientation")
                orientations = cori_stream.data  # Quaternions (w, x, y, z)
                # Interpolate if needed
                if len(cori_stream.timestamps) != len(timestamps):
                    orientations = self._interpolate_to_timestamps(
                        orientations, cori_stream.timestamps, timestamps
                    )
            except (KeyError, AttributeError):
                pass

            logger.info(f"Extracted IMU data with {len(timestamps)} samples")

            return IMUData(
                accelerometer=accel_data.astype(np.float32),
                gyroscope=gyro_data.astype(np.float32),
                timestamps=timestamps.astype(np.float64),
                gravity_vectors=gravity_vectors,
                orientations=orientations,
            )

        except Exception as e:
            logger.warning(f"Failed to extract IMU data: {e}")
            return None

    def _stream_to_xyz(self, stream) -> Tuple[np.ndarray, np.ndarray]:
        """Get (x, y, z) data and timestamps from a telemetry stream."""
        try:
            df = stream.to_dataframe()
        except Exception:
            return stream.data, stream.timestamps

        timestamps = (
            df["timestamp"].values.astype(np.float64)
            if "timestamp" in df.columns
            else stream.timestamps
        )

        axis_cols = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in {"x", "y", "z"}:
                axis_cols[col_lower] = col

        if len(axis_cols) == 3:
            data = df[[axis_cols["x"], axis_cols["y"], axis_cols["z"]]].values
            return data, timestamps

        value_cols = [col for col in df.columns if col != "timestamp"]
        if len(value_cols) >= 3:
            data = df[value_cols[:3]].values
            return data, timestamps

        return stream.data, stream.timestamps

    def _interpolate_to_timestamps(
        self,
        data: np.ndarray,
        src_timestamps: np.ndarray,
        dst_timestamps: np.ndarray,
    ) -> np.ndarray:
        """Interpolate data to new timestamps."""
        from scipy.interpolate import interp1d

        if data.ndim == 1:
            f = interp1d(
                src_timestamps,
                data,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            return f(dst_timestamps)
        else:
            # Interpolate each column
            result = np.zeros((len(dst_timestamps), data.shape[1]))
            for i in range(data.shape[1]):
                f = interp1d(
                    src_timestamps,
                    data[:, i],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                result[:, i] = f(dst_timestamps)
            return result

    def _extract_gps_only(self, video_path: Path) -> Optional[GPSTrack]:
        """Extract only GPS data (faster than full extraction)."""
        # Todo: clean this up, we extract GPS basically twice in extract_gps_imu and extract_initial_gps
        gps, _ = self.extract_gps_imu(video_path)
        return gps

    def extract_initial_gps(
        self, video_path: Path
    ) -> Optional[Tuple[float, float, float]]:
        """
        Extract just the initial GPS coordinates.

        Falls back to EXIF if gopro-py is not available.

        Returns:
            Tuple of (latitude, longitude, altitude) or None
        """
        # Try gopro-py first
        gps = self._extract_gps_only(video_path)
        if gps is not None and len(gps) > 0:
            valid_mask = (gps.latitudes != 0) & (gps.longitudes != 0)
            if valid_mask.any():
                first_idx = int(np.flatnonzero(valid_mask)[0])
                alt = gps.altitudes[first_idx] if gps.altitudes is not None else 0.0
                return (gps.latitudes[first_idx], gps.longitudes[first_idx], alt)

        # Fall back to EXIF
        return self._extract_gps_from_exif(video_path)

    def _extract_gps_from_exif(
        self, video_path: Path
    ) -> Optional[Tuple[float, float, float]]:
        """Extract GPS from video EXIF data using exiftool."""
        try:
            import exiftool
            import json

            with exiftool.ExifTool() as et:
                output = et.execute(b"-j", str(video_path).encode("utf-8"))
                metadata = json.loads(output)

                if not metadata:
                    return None

                exif = metadata[0]

                # Try QuickTime GPS coordinates first
                if "QuickTime:GPSCoordinates" in exif:
                    coords = exif["QuickTime:GPSCoordinates"]
                    if coords and coords != "N/A":
                        parts = coords.split()
                        if len(parts) >= 2:
                            lat = float(parts[0])
                            lon = float(parts[1])
                            alt = float(parts[2]) if len(parts) >= 3 else 0.0
                            return (lat, lon, alt)

                # Try individual fields
                lat = None
                lon = None
                alt = 0.0

                for field in ["Composite:GPSLatitude", "GPS:GPSLatitude"]:
                    if field in exif and exif[field] != "N/A":
                        lat = float(exif[field])
                        break

                for field in ["Composite:GPSLongitude", "GPS:GPSLongitude"]:
                    if field in exif and exif[field] != "N/A":
                        lon = float(exif[field])
                        break

                for field in ["Composite:GPSAltitude", "GPS:GPSAltitude"]:
                    if field in exif and exif[field] != "N/A":
                        alt = float(exif[field])
                        break

                if lat is not None and lon is not None:
                    return (lat, lon, alt)

        except Exception as e:
            logger.debug(f"EXIF GPS extraction failed: {e}")

        return None
