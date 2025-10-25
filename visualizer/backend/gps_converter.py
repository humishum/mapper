"""
GPS to Cartesian coordinate converter
"""

import math
from typing import Tuple, List, Dict


class GPSConverter:
    """Convert GPS coordinates to Cartesian coordinates for 3D visualization"""
    
    def __init__(self):
        self.reference_lat = None
        self.reference_lon = None
        self.reference_alt = None
    
    def set_reference_point(self, lat: float, lon: float, alt: float = 0.0):
        """
        Set the reference point for coordinate conversion
        
        Args:
            lat: Reference latitude
            lon: Reference longitude
            alt: Reference altitude
        """
        self.reference_lat = lat
        self.reference_lon = lon
        self.reference_alt = alt
    
    def gps_to_cartesian(
        self, 
        lat: float, 
        lon: float, 
        alt: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Convert GPS coordinates to Cartesian coordinates
        
        Uses simple equirectangular projection for small areas.
        For more accuracy over larger distances, could implement
        more sophisticated projections (UTM, etc.)
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in meters
            
        Returns:
            (x, y, z) in meters
        """
        # Set reference on first call
        if self.reference_lat is None:
            self.set_reference_point(lat, lon, alt)
            return (0.0, 0.0, 0.0)
        
        # Earth radius in meters
        R = 6371000
        
        # Convert to radians
        lat1 = math.radians(self.reference_lat)
        lon1 = math.radians(self.reference_lon)
        lat2 = math.radians(lat)
        lon2 = math.radians(lon)
        
        # Equirectangular approximation
        x = R * (lon2 - lon1) * math.cos((lat1 + lat2) / 2)
        z = R * (lat2 - lat1)
        y = alt - (self.reference_alt or 0.0)
        
        return (x, y, z)
    
    def batch_convert(
        self, 
        coordinates: List[Tuple[float, float, float]]
    ) -> List[Dict[str, float]]:
        """
        Convert multiple GPS coordinates to Cartesian
        
        Args:
            coordinates: List of (lat, lon, alt) tuples
            
        Returns:
            List of {x, y, z} dictionaries
        """
        results = []
        for lat, lon, alt in coordinates:
            x, y, z = self.gps_to_cartesian(lat, lon, alt)
            results.append({'x': x, 'y': y, 'z': z})
        return results

