# this file takes in set of pointcloud files and aligns them using the colored ICP algorithm 
# this dones't take into account the global aligment, just the single "folder" aligmnet 



from pathlib import Path 
import logging 
import open3d as o3d 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Aligner:
    def __init__(self, folder_path:Path, output_path:Path):
        self.folder_path = folder_path
        self.pointcloud_files = list(self.folder_path.glob("*.ply"))
        self.output_path = output_path
        self.threshold = 0.02
        self.voxel_size = 0.05

    def save_aligned_point_clouds(self, aligned_point_clouds:list[o3d.geometry.PointCloud]):
        for i, aligned_point_cloud in enumerate(aligned_point_clouds):
            o3d.io.write_point_cloud(self.output_path / f"{i}.ply", aligned_point_cloud)
    
    def align(self, method:str = "sequential_icp")->list[o3d.geometry.PointCloud]:
        point_clouds = [o3d.io.read_point_cloud(ply_file) for ply_file in self.pointcloud_files]

        if method == "sequential_icp": 
            aligned_point_clouds = self.align_sequential_icp(point_clouds)
        elif method == "colored_icp":
            aligned_point_clouds = self.align_colored_icp(point_clouds)
        else:
            raise NotImplementedError(f"Invalid method: {method}")

        return aligned_point_clouds
            

    def align_sequential_icp(self, point_clouds:list[o3d.geometry.PointCloud])->list[o3d.geometry.PointCloud]:
        # align the pointclouds sequentially using the ICP algorithm 
        for i in range(len(point_clouds) - 1):
            source = point_clouds[i]
            target = point_clouds[i + 1]
            # do global registration first
            source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(self.voxel_size, source, target)
            initial_global_registration = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, self.voxel_size)

            # then do ICP
            icp_result = o3d.pipelines.registration.registration_icp(
                source, target, self.threshold, initial_global_registration.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            point_clouds[i + 1] = icp_result.transformation @ point_clouds[i + 1]

            
        return point_clouds


    def align_colored_icp(self, point_clouds:list[o3d.geometry.PointCloud])->list[o3d.geometry.PointCloud]:
        raise NotImplementedError("Colored ICP is not implemented yet")


def preprocess_point_cloud(pcd, voxel_size):
    logger.info(f"Downsample with a voxel size {voxel_size}")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    logger.info(f"Estimate normal with search radius {radius_normal}")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    logger.info(f"Compute FPFH feature with search radius {radius_feature}")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, source,  target):
    logger.info("voxelize pointcloud")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result