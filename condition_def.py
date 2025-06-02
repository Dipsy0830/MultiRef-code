


def img_edit_tasks(gc, image_path):
    """
    定义通用的任务集合。
    """
    return {
        # "crop_path": lambda: gc.generate_extrapolation(image_path),
        # "low_res_path": lambda: gc.generate_super_res(image_path),
        # "low_light_path": lambda: gc.generate_light_enhance(image_path),
        # "lowres_crop_path": lambda: gc.generate_super_res_extrapolation(image_path),
        # "lowlight_crop_path": lambda: gc.generate_light_enhance_extrapolation(image_path),
        # "lowres_lowlight_path": lambda: gc.generate_super_res_light_enhance(image_path),
        
    }

def common_tasks(gc, image_path):
    """
    定义通用的任务集合。
    """
    return {
            "canny_path": lambda: gc.generate_canny(image_path),
            "depth_path": lambda: gc.generate_depth(image_path),
            "sketch_path": lambda: gc.generate_sketch(image_path),
            "semantic_map_path": lambda: gc.generate_semantic_map(image_path),
    }