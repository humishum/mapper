from must3r.model import MUSt3R, MEMORY_MODES, load_model
from must3r.model.blocks.attention import toggle_memory_efficient_attention
from must3r.demo.gradio import get_reconstructed_scene, get_3D_model_from_scene
from pathlib import Path
import os
import pickle
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class MuSt3RWrapper:
 
    """
    Wrapper for the MuSt3R model
    """

    def __init__(self, image_dir:Path, output_dir:Path, weights_path:Path ,retrieval_path:Path, image_size:int):
        
        self.image_dir = image_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.image_size = image_size
        self.weights_path = weights_path
        self.retrieval_path = retrieval_path
        self.model = None

        toggle_memory_efficient_attention(enabled=True)


    
    def load_model(self):
        self.model = load_model(self.weights_path, encoder=None, decoder=None, device='cuda',
                       img_size=self.image_size, memory_mode=None)

    def run(self):
        """
        Run the reconstruction process using the predefined parameters.
        Outputs PLY files for the reconstructed scene.
        """
        if self.model is None:
            self.load_model()
        
        # Get sorted list of images from the image directory
        images = sorted([os.path.join(self.image_dir, f)
                        for f in os.listdir(self.image_dir)
                        if os.path.isfile(os.path.join(self.image_dir, f))])
        
        # Default parameters (matching get_reconstruction.py defaults)
        num_mem_imgs = min(50, len(images))
        min_conf_thr = 1.05
        cam_size = 0.05
        execution_mode = "linseq"
        camera_conf_thr = 0.0
        num_refinements_iterations = 0
        logger.info(f"Image Path: {self.image_dir}")
        logger.info(f"Running reconstruction with {len(images)} images")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Retrieval path: {self.retrieval_path}")
        logger.info(f"Weights path: {self.weights_path}")
        logger.info(f"Image size: {self.image_size}")
        logger.info(f"Min confidence threshold: {min_conf_thr}")
        logger.info(f"Camera size: {cam_size}")
        logger.info(f"Execution mode: {execution_mode}")
        logger.info(f"Number of memory images: {num_mem_imgs}")
        logger.info(f"Number of refinements iterations: {num_refinements_iterations}")
        logger.info(f"Number of images: {len(images)}")
        logger.info(f"Image size: {self.image_size}")
        logger.info(f"Weights path: {self.weights_path}")
        logger.info(f"Retrieval path: {self.retrieval_path}")
        # Run reconstruction
        scene, outfile = get_reconstructed_scene(
            outdir=self.output_dir, 
            viser_server=None, 
            should_save_glb=True,
            model=self.model,
            retrieval=self.retrieval_path, 
            device='cuda',
            verbose=True, 
            image_size=self.image_size, 
            amp=False,
            filelist=images, 
            min_conf_thr=min_conf_thr,
            as_pointcloud=True, 
            transparent_cams=False, 
            local_pointmaps=False,
            cam_size=cam_size, 
            num_mem_images=num_mem_imgs, 
            max_bs=1,
            render_once=False, 
            camera_conf_thr=camera_conf_thr,
            num_refinements_iterations=num_refinements_iterations,
            execution_mode=execution_mode,
            vidseq_local_context_size=0, 
            keyframe_interval=3,
            slam_local_context_size=0,
            subsample=2, 
            min_conf_keyframe=1.5,
            keyframe_overlap_thr=0.05, 
            overlap_percentile=85
        )
        
        # Generate PLY file with confidence threshold of 6.0
        thresholds = [6.0, 5.0, 4.0, 3.0,  2.0, 1.5, min_conf_thr]
        for thr in thresholds:

            try:
                logger.info(f"Generating PLY file with confidence threshold {thr}")
                outfile = get_3D_model_from_scene(
                    outdir=self.output_dir, 
                    verbose=True, 
                    scene=scene, 
                    min_conf_thr=thr,
                    as_pointcloud=True, 
                    transparent_cams=False, 
                    cam_size=cam_size,
                    filename=f'scene_thr{thr}.ply'
                )
            except Exception as e:
                logger.error(f"Error generating PLY file with confidence threshold {thr}: {e}")
                pass
        
        # Save scene as pickle file
        with open(os.path.join(self.output_dir, 'scene.pkl'), 'wb') as f:
            pickle.dump(scene, f)
        
        return scene


if __name__ == "__main__":
    print(MEMORY_MODES)