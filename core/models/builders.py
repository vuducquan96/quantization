from pdb import set_trace as bp
from core.models.registry import LOSS, MODELS
from core.models.utils import VoxelGenerator
# from spconv.utils import PointToVoxel as VoxelGenerator
from core.loss.loss_lib import (WeightedSmoothL1LocalizationLoss,
                                SigmoidFocalClassificationLoss)


def model_builder(model_config, data_config, box_coders=None, anchor_box_function=None):

    voxel_generator = VoxelGenerator(
        voxel_size=data_config["VOXEL_GENERATOR"]["VOXEL_SIZE"],
        point_cloud_range=data_config["POINT_CLOUD_RANGE"],
        # num_point_features=3, 
        max_num_points_per_voxel=data_config["VOXEL_GENERATOR"]["MAX_NUMBER_OF_POINTS_PER_VOXEL"],
        max_num_voxels=data_config["VOXEL_GENERATOR"]["MAX_VOXELS"]
    )

    config_box = {
        "model_config": model_config,
        "data_config": data_config,
        "loc_loss": LOSS.get(model_config.LOSS.REG_LOSS.name)(),
        "cls_loss": LOSS.get(model_config.LOSS.CLS_LOSS.name)(),
        "box_coders": box_coders,
        "target_assigner": anchor_box_function,
        "grid_size": voxel_generator.grid_size}
    
    model = MODELS.get(model_config.base)(config_box)

    return model
