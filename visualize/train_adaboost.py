import sys
import os
import re
from classes import *


def get_filepaths(directory):
    """
    retrieves a list of all filepaths from this directory
    """
    output = []

    onlyfiles = [
        f
        for f in os.listdir(os.path.realpath(directory))
        if os.path.isfile(os.path.join(os.path.realpath(directory), f))
    ]

    for k in onlyfiles:
        output.append(os.path.realpath(os.path.join(directory, k)))

    return output

def create_max_projections():
    filepaths = get_filepaths("D:/590Data/")
    for k in filepaths:
        if k.endswith(".a3d"):
            print("Creating projection for {}".format(k))
            pid = re.search(r"(.+)\.a3d", k).group(1)
            if (os.path.realpath(pid + "_projection.png") in filepaths):
                continue
            bs = BodyScan(k)
            bs.create_max_projection(k)
        else:
            continue


def main(argv):
    pos_dir = "../data/positive_examples"
    neg_dir = "../data/negative_examples"
    with open("ada_output.txt", "w") as f:
        pass
    ab = SCAdaBoost(
        "../data/stage1_labels.csv",
        pos_dir,
        neg_dir,
        "../data/negative_examples/example.png"
    )
    ab.load_examples(pos_dir, neg_dir, 175, 225)
    ab.get_feature_vals_for_all()
    del ab.x
    ab.adaboost(10000, True)


if __name__ == "__main__":
    main(sys.argv)
