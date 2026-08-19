import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pacmap
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from umap import UMAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="tsne")
    parser.add_argument("--target", type=str, default="age_at_visit")
    parser.add_argument("--dict", type=str)
    args = parser.parse_args()

    visu_method = args.method
    target_key = args.target
    dict_path = args.dict

    with open(dict_path, "rb") as f:
        data_dict = pickle.load(f)

    x = data_dict["x0"]
    y = np.asarray(data_dict[target_key])

    print(
        f"Visualizing {x.shape[0]} samples with {x.shape[1]} features using {visu_method} for target '{target_key}'."
    )

    seed = None

    if visu_method == "umap":
        visu_model = UMAP(random_state=seed)
    elif visu_method == "pacmap":
        visu_model = pacmap.PaCMAP()
    elif visu_method == "tsne":
        visu_model = TSNE(random_state=seed)
    else:
        raise ValueError(f"Unknown method: {visu_method}")

    embedding = visu_model.fit_transform(x)

    plt.figure(figsize=(8, 6))

    # Check whether y is categorical
    is_categorical = (
        y.dtype.kind in {"U", "S", "O"}  # strings/objects
        or len(np.unique(y)) < 20  # small number of unique values
    )

    if is_categorical:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=y_encoded,
            # cmap="tab20",
            # cmap="coolwarm",
            cmap="winter",
            # cmap="Set1",
            # cmap="Dark2",
            alpha=0.8,
            s=10,
        )

        handles = []
        for i, label in enumerate(le.classes_):
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    color=scatter.cmap(scatter.norm(i)),
                    label=str(label),
                )
            )

        plt.legend(
            handles=handles,
            title=target_key,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

    else:
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=y,
            cmap="Spectral",
            alpha=0.8,
            s=10,
        )
        plt.colorbar(scatter, label=target_key)

    plt.title(f"{visu_method.upper()} projection of features")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()

    # save as svg in the same folder as the dict_path
    output_path = dict_path.replace(".pkl", f"_{visu_method}_{target_key}.svg")
    plt.savefig(output_path, format="svg", bbox_inches="tight")

    output_path = dict_path.replace(".pkl", f"_{visu_method}_{target_key}.png")
    plt.savefig(output_path, format="png", bbox_inches="tight")

    plt.show()

    plt.close()
