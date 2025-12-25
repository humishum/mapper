"""FastAPI based server to serve pointcloud data """

import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import PORT, DATA_DIR, DEFAULT_THRESHOLD, MAX_POINTS
from .data_service import PointCloudDataService

# logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="3D Pointcloud Viewer API",
    description="API for loading and serving pointcloud data",
    version="0.1.0",
)

# CORS to talk to react
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # TODO: flag this for prod!!!! 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data service
data_broker = PointCloudDataService(DATA_DIR)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "3D Pointcloud Viewer API",
        "version": "0.1.0",
        "endpoints": {
            "locations": "/api/locations",
            "pointcloud": "/api/pointcloud/{location_name}",
        },
    }


@app.get("/api/locations")
async def get_locations():
    """
    Get all available pointcloud locations.

    Returns:
        List of locations with GPS coordinates and metadata
    """
    try:
        locations = data_broker.get_all_locations()
        return {
            "count": len(locations),
            "locations": [loc.to_dict() for loc in locations],
        }
    except Exception as e:
        logger.error(f"Error getting locations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pointcloud/{location_name}")
async def get_pointcloud(
    location_name: str,
    sequence: str = Query(default="sequence_1", description="Sequence name"),
    threshold: float = Query(
        default=DEFAULT_THRESHOLD, ge=0.5, le=10.0, description="Threshold value"
    ),
    max_points: int = Query(
        default=MAX_POINTS, ge=1000, le=500000, description="Maximum points to return"
    ),
    use_gps_coords: bool = Query(
        default=True, description="Convert to GPS coordinates"
    ),
):
    """
    Load and return pointcloud data for a specific location.

    Args:
        location_name: Name of the location
        sequence: Sequence name (default: sequence_1)
        threshold: Threshold value for PLY file selection (default: 2.0)
        max_points: Maximum number of points to return (default: 100000)
        use_gps_coords: Convert to GPS coordinates (default: True)

    Returns:
        Pointcloud data with positions and colors
    """
    try:
        logger.info(
            f"Loading pointcloud: {location_name}, sequence={sequence}, threshold={threshold}, max_points={max_points}"
        )

        pointcloud = data_broker.load_pointcloud(
            location=location_name,
            sequence=sequence,
            threshold=threshold,
            max_points=max_points,
            use_gps_coords=use_gps_coords,
        )

        if pointcloud is None:
            raise HTTPException(
                status_code=404,
                detail=f"Pointcloud not found for location: {location_name}",
            )

        return pointcloud

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading pointcloud: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "data_dir": str(DATA_DIR)}


def main():
    """Run the server."""
    logger.info(f"Starting server on port {PORT}")
    logger.info(f"Data directory: {DATA_DIR}")

    uvicorn.run("backend.server:app", host="0.0.0.0", port=PORT, reload=True)


if __name__ == "__main__":
    main()
