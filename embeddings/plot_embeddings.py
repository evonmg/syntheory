import argparse
from json import loads
import numpy as np
import pandas as pd
import zarr
import umap

from config import OUTPUT_DIR, load_config
from probe.main import _is_equal_model_types
from probe.probe_config import CONCEPT_LABELS

class EmbeddingsPlot:

    def load_embeddings(
            self,
            dataset_labels_filepath,
            dataset_settings,
            embeddings_zarr_filepath,
            model_type
        ) -> None:
        self.dataset_labels_filepath = dataset_labels_filepath
        self.label_columns = [label_column for (_, label_column) in dataset_settings]
        self.model_type = model_type

        # load the dataset
        dataset_labels = pd.read_csv(dataset_labels_filepath)

        self.dataset_labels = dataset_labels

        # add the zarr idx if not listed explicitly
        if "zarr_idx" not in set(dataset_labels.columns):
            dataset_labels["zarr_idx"] = np.arange(dataset_labels.shape[0])

        # load the zarr, read only mode
        data = zarr.open(embeddings_zarr_filepath, mode="r")

        self.is_foundation_model_layers = len(data.shape) == 3
        self.embeddings = data

        selector = self.dataset_labels["zarr_idx"].to_numpy()
        if self.is_foundation_model_layers:
            # get last layer
            self.X = np.array(self.embeddings[selector][:, -1, :], dtype=np.float32)
        else:
            # handcrafted features or encoders (only one layer)
            self.X = np.array(self.embeddings[selector], dtype=np.float32)

        self.ys = [self.dataset_labels[label] for label in self.label_columns]

    def plot_umap(self) -> None:
        # loop through label columns
        for i in range(len(self.label_columns)):
            # aggregate all the dataset label splits to plot
            print(f"plotting UMAP for {self.label_columns[i]} in {self.model_type}")

            # cosine similarity for word embeddings
            mapper = umap.UMAP(n_neighbors=50, min_dist=0.1, metric="cosine").fit(self.X)
            ax = umap.plot.points(mapper, labels=self.ys[i])

            ax.get_figure().savefig(f"umap_plot_{str(self.dataset_labels_filepath).split("/")[6]}_{self.label_columns[i]}_{self.model_type}.png")

            print("done")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()
    config = load_config(args.config)

    concepts = config['concepts']
    conds = config['conditionings']
    models = config['models']

    for concept_name in concepts:
        for model in models:
            if model.split("_LM_")[-1] != "L" and model != "MUSICGEN_TEXT_ENCODER":
                continue

            base_path= OUTPUT_DIR / concept_name

            dataset_settings = CONCEPT_LABELS[concept_name]

            # TODO: fix so that we can do both text and audio
            if "text" in conds:
                dataset_info = base_path / "prompts.csv"
            elif "audio" in conds:
                dataset_info = base_path / "info.csv"

            zarr_files = list(base_path.glob("*.zarr"))

            model_embeddings_infos = []

            # loop through all zarr files
            for z in zarr_files:
                f_name = z.parts[-1]

                if not f_name.startswith(concept_name):
                    # expect embeddings arrays to be prefixed with the concept name
                    continue

                # get the settings associated with the model that produced these embeddings
                model_hash = f_name.split(".zarr")[0].split("_")[-1]
                model_settings_path = base_path / (concept_name + "_" + model_hash + ".json")
                model_settings = loads(model_settings_path.read_text())

                model_name = model_settings["model_name"]

                emb = zarr.open(z)

                model_name = model_settings["model_name"]
                if model_name == "JUKEBOX":
                    model_type = "JUKEBOX"
                elif "MUSICGEN_DECODER" in model_name:
                    model_type = "MUSICGEN_DECODER"
                elif "MUSICGEN_AUDIO_ENCODER" in model_name:
                    model_type = "MUSICGEN_AUDIO_ENCODER"
                elif "MUSICGEN_TEXT_ENCODER" in model_name:
                    model_type = "MUSICGEN_TEXT_ENCODER"
                else:
                    # handcrafted features, default to L
                    model_type = model_settings["model_name"]

                exp_info = {
                    "zarr_filepath": z,
                    "dataset_labels_path": dataset_info,
                    "model_type": model_type,
                }
                model_embeddings_infos.append(exp_info)
        
            # look up the correct location for the embeddings given experiment config
            exp_info = None
            for e in model_embeddings_infos:
                if (
                    _is_equal_model_types(e["model_type"], model)
                ):
                    exp_info = e
            
            emb = EmbeddingsPlot()
            emb.load_embeddings(
                dataset_labels_filepath=exp_info["dataset_labels_path"],
                dataset_settings=dataset_settings,
                embeddings_zarr_filepath=exp_info["zarr_filepath"],
                model_type=exp_info["model_type"]
            )

            emb.plot_umap()

if __name__ == "__main__":
    main()