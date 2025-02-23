import os
import json
from PIL import Image

def labelme_to_coco(labelme_folder, output_json):
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    category_dict = {}
    annotation_id = 1

    # Add categories
    categories = [{"id": 0, "name": "empty"}, {"id": 1, "name": "parked"}]
    for category in categories:
        coco["categories"].append(category)
        category_dict[category["name"]] = category["id"]

    # Process each Labelme JSON file
    for json_file in os.listdir(labelme_folder):
        if json_file.endswith(".json"):
            with open(os.path.join(labelme_folder, json_file), 'r') as f:
                data = json.load(f)

            # Add image info
            image_id = len(coco["images"]) + 1
            image = {
                "id": image_id,
                "file_name": data["imagePath"],
                "width": data["imageWidth"],
                "height": data["imageHeight"]
            }
            coco["images"].append(image)

            # Add annotations
            for shape in data["shapes"]:
                if shape["shape_type"] == "polygon":
                    points = shape["points"]
                    segmentation = [coord for point in points for coord in point]
                    
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    xmin, ymin = min(x_coords), min(y_coords)
                    xmax, ymax = max(x_coords), max(y_coords)
                    width, height = xmax - xmin, ymax - ymin

                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_dict[shape["label"]],
                        "segmentation": [segmentation],
                        "bbox": [xmin, ymin, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    }
                    coco["annotations"].append(annotation)
                    annotation_id += 1

    # Save to COCO JSON
    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=4)

# Run the function
labelme_folder = r"D:\yolo\dataset\labels\train"  # Đường dẫn tới folder chứa JSON Labelme
output_json = "output_coco_train.json"  # Tên file output COCO
labelme_to_coco(labelme_folder, output_json)
