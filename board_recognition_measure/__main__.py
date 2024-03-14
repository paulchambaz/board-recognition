import json
import os
import pathlib
import argparse

import matplotlib.image as mimg
import matplotlib.pyplot as mplt
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import board_recognition as br


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure performance of board-recognition")
    parser.add_argument("-s", "--save", action='store_true', help="Save the images or not (slower)")
    args = parser.parse_args()

    save = args.save

    board_training = measure_performance("ground-truth/board/training", br.get_board, save)
    board_test = measure_performance("ground-truth/board/test", br.get_board, save)

    final_data = {
        "training": {
            "board": board_training,
            "test": None,
        },
        "test": {
            "board": board_test,
            "test": None,
        },
    }

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    files = [f for f in os.listdir(results_dir) if f.endswith(".json")]

    previous_data = None
     
    max_num = 0

    if files:
        max_num = max(int(f.split('.')[0]) for f in files)

        latest_file_name = f"{results_dir}/{max_num}.json"
        with open(latest_file_name, "r") as f:
            previous_data = json.load(f)

    new_file_num = max_num + 1
    new_file_name = f"{results_dir}/{new_file_num}.json"

    with open(new_file_name, "w") as f:
        json.dump(final_data, f, indent=2)

    if previous_data:
        current_fitness_board = final_data["training"]["board"]["metadata"]["fitness"]
        previous_fitness_board = previous_data["training"]["board"]["metadata"]["fitness"]
        diff = current_fitness_board - previous_fitness_board
        if diff >= 0:
            print(f"Board: +{diff}")
        else:
            print(f"Board: {diff}")

def measure_performance(path, processing_function, save_images=False):
    files = load_files(path)

    values = []
    for file in tqdm(files):
        n = file.stem
        image_file = f"data/{n}.jpg"
        image = load_image(image_file)

        ground_truth = load_polygons_from_json(file)
        ground_truth = np.array(ground_truth[0]).astype(int)
        image_ground_truth = create_polygon_image(image, ground_truth)

        result, _ = processing_function(image)
        image_result = create_polygon_image(image, result)

        true_positive = np.logical_and(
            image_ground_truth == 255, image_result == 255
        ).sum()
        false_negative = np.logical_and(
            image_ground_truth == 255, image_result == 0
        ).sum()
        false_positive = np.logical_and(
            image_ground_truth == 0, image_result == 255
        ).sum()
        true_negative = np.logical_and(image_ground_truth == 0, image_result == 0).sum()

        total = true_positive + false_negative + false_positive + true_negative
        success = (true_positive + true_negative) / total

        true_positive = true_positive / total
        false_negative = false_negative / total
        false_positive = false_positive / total
        true_negative = true_negative / total

        value = {
            "name": n,
            "success": success,
            "true_positive": true_positive,
            "false_negative": false_negative,
            "false_positive": false_positive,
            "true_negative": true_negative,
        }
        values.append(value)

        if save_images:
            save_image(image, image_ground_truth, image_result, value, f"results/{path.replace('/', '-')}-{n}.jpg")

    values = sorted(values, key=lambda x: x["success"])

    success_rates = [v["success"] for v in values]
    mean_success = np.mean(success_rates)
    median_success = np.median(success_rates)
    std_dev_success = np.std(success_rates)
    worst_success = values[0]["success"]
    best_success = values[-1]["success"]

    mean_false_positive = np.mean([v["false_positive"] for v in values])
    mean_false_negative = np.mean([v["false_negative"] for v in values])

    fitness = (9 * median_success + mean_success) / 10

    metadata = {
        "fitness": fitness,
        "median": median_success,
        "mean": mean_success,
        "std_dev": std_dev_success,
        "worst": worst_success,
        "best": best_success,
        "mean_false_positives": mean_false_positive,
        "mean_false_negatives": mean_false_negative,
    }

    final_dict = {
        "metadata": metadata,
        "values": values,
    }

    return final_dict


def load_files(path):
    path = pathlib.Path(path)
    return [file for file in path.iterdir() if file.is_file()]


def load_image(path):
    return mimg.imread(path)


def load_polygons_from_json(json_file_path):
    polygons = []
    try:
        f = open(json_file_path, "r")
        json_data = json.load(f)
    except OSError:
        return polygons

    for shape in json_data["shapes"]:
        polygons.append(shape["points"])
    return polygons


def create_polygon_image(image, polygons):
    image_pil = Image.fromarray(image)
    size = image_pil.size

    image_polygon = Image.new("L", size)
    draw = ImageDraw.Draw(image_polygon)

    if not isinstance(polygons[0], list):
        polygons = [polygons]

    for poly in polygons:
        points = [tuple(point) for point in poly]
        draw.polygon(points, fill="white")

    return np.array(image_polygon)


def save_image(original_image, image_ground_truth, image_result, value, file):
    fig, axs = mplt.subplots(2, 2, figsize=(10, 10))
    
    axs[0, 0].imshow(original_image)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    axs[1, 0].imshow(image_ground_truth, cmap='gray')
    axs[1, 0].set_title('Ground Truth')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(image_result, cmap='gray')
    axs[1, 1].set_title('Result')
    axs[1, 1].axis('off')

    overlay = np.zeros_like(original_image)

    overlay[(image_ground_truth == 0) & (image_result == 0)] = [0, 0, 0]
    overlay[(image_ground_truth == 255) & (image_result == 0)] = [0, 255, 255]
    overlay[(image_ground_truth == 0) & (image_result == 255)] = [255, 0, 0]
    overlay[(image_ground_truth == 255) & (image_result == 255)] = [255, 255, 255]

    axs[0, 1].imshow(original_image)
    axs[0, 1].imshow(overlay, alpha=0.5)
    axs[0, 1].set_title('Overlay Comparison')
    axs[0, 1].axis('off')

    success = int(100 * value['success'])
    false_positive = int(100 * value['false_positive'])
    false_negative = int(100 * value['false_negative'])
    fig.suptitle(f"Success: ~{success}%, False Positive: ~{false_positive}%, False Negative: ~{false_negative}%", fontsize=16)
    
    mplt.tight_layout(rect=[0, 0.03, 1, 0.95])
    mplt.savefig(file, dpi=300)
    mplt.close()

if __name__ == "__main__":
    main()
