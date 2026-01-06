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

    def __init__(self):
        """Initialize telemetry extractor."""
        self._gopropy_available = None
        self._check_gopropy()

    def _check_gopropy(self) -> bool:
        """Check if gopro-py is available."""
        if self._gopropy_available is not None:
            return self._gopropy_available

        try:
            import gopropy
            from gopropy import StreamNotFoundError

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
            print(f"Telemetry: {telemetry.list_streams()}")
            gps_track = self._extract_gps(telemetry)
            imu_data = self._extract_imu(telemetry)

            return gps_track, imu_data

        except Exception as e:
            logger.warning(f"Failed to extract telemetry: {e}")
            return None, None

    def _extract_gps(self, telemetry) -> Optional[GPSTrack]:
        """Extract GPS track from telemetry."""
        # todo: clean up this code, super weird
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
                "GPS (Lat., Long., Alt., 2D speed, 3D speed)",
            ]:
                try:
                    print(f"Trying to get GPS stream: {name}")
                    gps_stream = telemetry.get_stream(name)
                    print(f"Found GPS stream: {gps_stream}")
                    break
                except (KeyError, AttributeError, StreamNotFoundError) as e:
                    print(f"Failed to get GPS stream: {name}")
                    print(f"Error: {e}")
                    continue

            if gps_stream is None:
                print("No GPS stream found in telemetry")
                return None

            # Convert to dataframe for easier access
            df = gps_stream.to_dataframe()

            # Find latitude/longitude columns
            lat_col = None
            lon_col = None
            alt_col = None

            for col in df.columns:
                col_lower = col.lower()
                if "lat" in col_lower:
                    lat_col = col
                elif "lon" in col_lower:
                    lon_col = col
                elif "alt" in col_lower:
                    alt_col = col

            if lat_col is None or lon_col is None:
                print("Could not find lat/lon columns in GPS data")
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

            # Filter out invalid GPS readings (0,0 coordinates)
            valid_mask = (
                np.isfinite(latitudes)
                & np.isfinite(longitudes)
                & (latitudes != 0)
                & (longitudes != 0)
            )
            if not valid_mask.any():
                print("No valid GPS readings found (all 0,0)")
                return None

            latitudes = latitudes[valid_mask]
            longitudes = longitudes[valid_mask]
            if altitudes is not None:
                altitudes = altitudes[valid_mask]
            if timestamps is not None:
                timestamps = timestamps[valid_mask]

            print(f"Extracted GPS track with {len(latitudes)} points")

            return GPSTrack(
                latitudes=latitudes,
                longitudes=longitudes,
                altitudes=altitudes,
                timestamps=timestamps,
            )

        except Exception as e:
            logger.warning(f"Failed to extract GPS data: {e}")
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
