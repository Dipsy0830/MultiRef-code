# Synthetic Eval Pipeline

> **Note:** All external dependencies must be installed manually. The evaluation code expects these directories and files to exist at the specified relative paths.

## Usage

```bash
python process_for_eval.py --input <input_json_path> --output <output_json_path> --workers 10
```
- `--input`: Input JSON file, must contain image paths and related info
- `--output`: Output JSON file
- `--workers`: Number of parallel processes


**Input JSON Format**

The `--input` argument should point to a JSON file containing a list of entries.  
**Each entry must follow the structure below (fields can be extended as needed):**

```json
{
    "original_image_path": "<path to original image>",
    "conditions": {
        "semantic_map_path": "<path to semantic map image>",
        "sketch_path": "<path to sketch image>",
        "canny_path": "<path to canny image>",
        "bbox_path": "<path to bbox image>",
        "depth_path": "<path to depth image>",
        "mask_path": "<path to mask image>",
        "caption": "<caption text>",
        "style_path": [
            "<path to style image 1>",
            "<path to style image 2>"
        ]
    },
    "instructions": {
        "conditions": [
            "mask",
            "caption"
        ],
        "image_path_mapping": {
            "mask_image": "<path to mask image>"
        },
        "original_prompts": "<original prompt text>",
        "enhance_prompts": [
            "<enhanced prompt text>"
        ]
    },
    "judge": {
        "Semantic-Map Alignment": 5,
        "Semantic-Map Quality": 5,
        "Sketch Alignment": 5,
        "Sketch Quality": 5,
        "Canny Alignment": 5,
        "Canny Quality": 5,
        "Bounding-Box Accuracy": 5,
        "Depth Alignment": 5,
        "Depth Quality": 5,
        "Mask Alignment": 5,
        "Caption Alignment": 5
    },
    "results": {
        "image_path": "<path to generated image>",
        "model_name": "<model name>",
        "other_info": {
            "guidance_scale": 2.5,
            "img_guidance_scale": 1.6,
            "seed": 0
        }
    }
}
```

- The JSON file should be a list of such entries (i.e., `[ {...}, {...}, ... ]`).
- All image paths can be either relative or absolute.
- The `conditions` field should include all the condition types you want to evaluate.
- The `results.image_path` should point to the generated image to be evaluated.

**Note:**  
Replace all `<...>` placeholders with your actual data. 