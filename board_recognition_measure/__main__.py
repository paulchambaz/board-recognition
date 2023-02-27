"""This file is used to measure the performance of our board detection
algorithms"""

import matplotlib.image as mimg

import pathlib
import statistics
import random
from tqdm import tqdm

def load_files(path):
    path = pathlib.Path("data")
    return [str(file) for file in path.iterdir() if file.is_file()]

def load_image(path):
    return  mimg.imread(path)

def main():
    files = load_files("data")
    scores = []

    for file in tqdm(files):
        image = load_image(file)
        scores.append(random.random())

    mean = statistics.mean(scores)
    median = statistics.median(scores)
    worst = min(scores)
    best = max(scores)

    print("Mean:", mean)
    print("Median:", median)
    print("Worst:", worst)
    print("Best:", best)
