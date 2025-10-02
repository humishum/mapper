# this file takes in set of pointcloud files and aligns them using the colored ICP algorithm 
# this dones't take into account the actual global aligment, just the single "folder" aligmnet 



from pathlib import Path 
import logging 
import open3d as o3d 
import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Aligner:
    def __init__(self, folder_path:Path, output_path:Path, threshold:float = 0.02, voxel_size:float = 0.1):
        self.folder_path = folder_path
        self.pointcloud_files = list(self.folder_path.glob("*.ply"))
        self.output_path = output_path
        self.threshold = threshold
        self.voxel_size = voxel_size
        self.aligned_point_clouds = []

        if not self.output_path.exists():
            self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.point_clouds = self.load_point_clouds()
        print(f" Input Folder: {self.folder_path}")
        print(f" Output Folder: {self.output_path}")
        print(f" Voxel Size: {self.voxel_size}")
        print(f" Threshold: {self.threshold}")
        print(f" Number of point clouds: {len(self.point_clouds)}")
        
    def load_point_clouds(self)->list[o3d.geometry.PointCloud]:
        return [o3d.io.read_point_cloud(ply_file) for ply_file in self.pointcloud_files]

    def save_aligned_point_clouds(self):
        if not self.aligned_point_clouds:
            raise ValueError("No aligned point clouds to save, run align() first")
        for i, aligned_point_cloud in enumerate(self.aligned_point_clouds):
            o3d.io.write_point_cloud(self.output_path / f"{i}.ply", aligned_point_cloud)
    
    def align(self, method:str = "sequential_icp")->list[o3d.geometry.PointCloud]:
        # TODO at some later point, implement proper factory pattern for this

        if method == "sequential_icp": 
            aligned_point_clouds = self._align_sequential_icp(self.point_clouds)
        elif method == "sequential_point_to_plane":
            aligned_point_clouds = self._align_sequential_point_to_plane(self.point_clouds)
        elif method == "colored_icp":
            aligned_point_clouds = self._align_colored_icp(self.point_clouds)
        else:
            raise NotImplementedError(f"Invalid method: {method}")

        self.aligned_point_clouds = aligned_point_clouds
        return aligned_point_clouds
            

    def _align_sequential_icp(self, point_clouds:list[o3d.geometry.PointCloud])->list[o3d.geometry.PointCloud]:
        # align the pointclouds sequentially using the ICP algorithm 
        for i in tqdm.tqdm(range(len(point_clouds) - 1)):
            source = point_clouds[i]
            target = point_clouds[i + 1]
            # do global registration first
            source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(self.voxel_size, source, target)
            initial_global_registration = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, self.voxel_size)

            # then do ICP
            icp_result = o3d.pipelines.registration.registration_icp(
                source, target, self.threshold, initial_global_registration.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            point_clouds[i + 1] = point_clouds[i + 1].transform(icp_result.transformation)

        return point_clouds

    def _align_sequential_point_to_plane(self, point_clouds:list[o3d.geometry.PointCloud])->list[o3d.geometry.PointCloud]:
        # align the pointclouds sequentially using the Point to Plane ICP algorithm
        for i in tqdm.tqdm(range(len(point_clouds) - 1)):
            source = point_clouds[i]
            target = point_clouds[i + 1]
            # do global registration first
            source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(self.voxel_size, source, target)
            initial_global_registration = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, self.voxel_size)

            source.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30),
                fast_normal_computation=True
            )
            target.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30),
                fast_normal_computation=True
            )
            # then do ICP
            icp_result = o3d.pipelines.registration.registration_icp(
                source, target, self.threshold, initial_global_registration.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            point_clouds[i + 1] = point_clouds[i + 1].transform(icp_result.transformation)

        return point_clouds


    def _align_colored_icp(self, point_clouds:list[o3d.geometry.PointCloud])->list[o3d.geometry.PointCloud]:
        # Align the pointclouds sequentially using the Colored ICP algorithm
        for i in tqdm.tqdm(range(len(point_clouds) - 1)):
            source = point_clouds[i]
            target = point_clouds[i + 1]

            # Preprocess: downsample and estimate normals
            source_down = source.voxel_down_sample(self.voxel_size)
            target_down = target.voxel_down_sample(self.voxel_size)

            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30),
                fast_normal_computation=True
            )
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30),
                fast_normal_computation=True
            )

            # Initial alignment using global registration (RANSAC)
            source_down_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                source_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=100)
            )
            target_down_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                target_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=100)
            )

            distance_threshold = self.voxel_size * 1.5
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_down_fpfh, target_down_fpfh, True,
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
            )

            source.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30),
                fast_normal_computation=True
            )
            target.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30),
                fast_normal_computation=True
            )

            # Colored ICP refinement
            icp_result = o3d.pipelines.registration.registration_colored_icp(
                source, target, self.threshold, result_ransac.transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                 relative_rmse=1e-6,
                                                                 max_iteration=50)
            )
            point_clouds[i + 1] = point_clouds[i + 1].transform(icp_result.transformation)

        return point_clouds


def preprocess_point_cloud(pcd, voxel_size):
    logger.info(f"Downsample with a voxel size {voxel_size} (points: {len(pcd.points)})")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    logger.info(f"After downsampling: {len(pcd_down.points)} points")

    radius_normal = voxel_size * 2
    logger.info(f"Estimate normal with search radius {radius_normal}")
    # Reduce max_nn from 30 to 15 for faster computation
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=15), fast_normal_computation=True)

    radius_feature = voxel_size * 5
    logger.info(f"Compute FPFH feature with search radius {radius_feature}")
    # Reduce max_nn from 100 to 50 for faster computation
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=50))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, source,  target):
    logger.info("voxelize pointcloud")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    logger.info(":: RANSAC registration on downsampled point clouds.")
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



# Multiprocessing alignment function

def run_alignment_wrapper(args):
    # Unpack arguments and call run_alignment with proper signature
    folder_path, output_path, threshold, voxel_size, method = args
    
    import time
    print(f"Starting alignment: folder={folder_path}, output={output_path}, voxel_size={voxel_size}, method={method}")
    time_start = time.time()
    
    try:
        aligner = Aligner(folder_path, output_path, threshold=threshold, voxel_size=voxel_size)
        print(f"Created aligner for method: {method}")
        aligner.align(method=method)
        print(f"Alignment complete for method: {method}")
        aligner.save_aligned_point_clouds()
        time_end = time.time()
        print(f"Time taken for method {method}: {time_end - time_start:.2f} seconds")
        return f"Success: method={method}, voxel_size={voxel_size}"
    except Exception as e:
        time_end = time.time()
        print(f"Time taken for method {method}: {time_end - time_start:.2f} seconds")
        print(f"Error with method {method}: {e}")
        return f"Error: method={method} - {e}"

if __name__ == "__main__":
    import multiprocessing as mp
    input_dir  = Path("/home/ape/repos/must3rdemo/must3r/test_dir")
    output_base = Path("/home/ape/repos/must3rdemo/must3r/test_dir/test_alignment")

    print("Testing multiprocessing with spawn method...")
    mp.set_start_method('spawn', force=True)

    # Define the alignment methods to test (single voxel size)
    alignment_methods = [ "sequential_point_to_plane", "colored_icp"]
    voxel_size = 0.1  # Use a single voxel size

    # Prepare argument tuples for each method
    tasks = []
    for method in alignment_methods:
        out_path = output_base / f"{method}_voxel{voxel_size}"
        tasks.append((
            input_dir,
            out_path,
            0.02,  # threshold (can be parameterized if needed)
            voxel_size,
            method
        ))

    with mp.Pool(processes=min(len(tasks), mp.cpu_count())) as pool:
        results = pool.map(run_alignment_wrapper, tasks)
        print(f"Multiprocessing results: {results}")
