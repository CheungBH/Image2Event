from datasets import GeneratorBasedBuilder, DatasetInfo, Split, SplitGenerator
from pathlib import Path
import json
import datasets

_DESCRIPTION = """\
Custom dataset for ControlNet training containing paired images with conditioning frames.
"""

_CITATION = """\
@InProceedings{,
  title = {Event Dataset},
  author = {Your Name},
  year = {2023}
}
"""

class EventDataset(GeneratorBasedBuilder):
    """Custom dataset for event-conditioned image generation."""

    VERSION = datasets.Version("1.1.0")
    DEFAULT_CONFIG_NAME = "default"
    BUILDER_CONFIGS = [datasets.BuilderConfig(name=DEFAULT_CONFIG_NAME, version=VERSION)]

    def _info(self):
        return DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            features=datasets.Features({
                "text": datasets.Value("string"),
                "image": datasets.Image(),
                "conditioning_image": datasets.Image(),
                "optical_flow": datasets.Value("string"),
            }),
            homepage="https://example.com/dataset",
            license="CC-BY-4.0",
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata_path": "train_metadata.jsonl"  # 直接使用文件名，不含子目录
                },
            )
        ]

    def _split_generators(self, dl_manager):
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "metadata_path": Path(self.config.data_dir) / "metadata.jsonl",
                    "base_dir": Path(self.config.data_dir)
                },
            )
        ]

    # def _split_generators(self, dl_manager):
    #     return [
    #         SplitGenerator(
    #             name=Split.TRAIN,
    #             gen_kwargs={
    #                 "metadata_path": Path(self.config.data_dir) / "metadata.jsonl",
    #                 "base_dir": Path(self.config.data_dir)
    #             },
    #         ),
        #     SplitGenerator(
        #         name=Split.VALIDATION,
        #         gen_kwargs={
        #             "metadata_path": Path(self.config.data_dir) / "validation.jsonl",
        #             "base_dir": Path(self.config.data_dir)
        #         },
        #     )
        #   ]

    def _generate_examples(self, metadata_path, base_dir):
        """Yields examples as (key, example) tuples."""
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    data = json.loads(line)

                    # Validate paths exist
                    img_path = base_dir / data["image"]
                    cond_path = base_dir / data["conditioning_image"]
                    oflow_path = base_dir / data["optical_flow"]
                    if not img_path.exists():
                        raise FileNotFoundError(f"Missing image: {img_path}")
                    if not cond_path.exists():
                        raise FileNotFoundError(f"Missing conditioning image: {cond_path}")

                    yield idx, {
                        "text": data["text"],
                        "image": str(img_path.resolve()),
                        "conditioning_image": str(cond_path.resolve()),
                        "optical_flow":  oflow_path,
                    }
        except Exception as e:
            raise RuntimeError(f"Error generating examples from {metadata_path}") from e

    # def _post_processing_resources(self, split):
    #     return {
    #         "metadata_file": str(Path(self.config.data_dir) / f"{split}_metadata.jsonl")
    #     }
    def _post_processing_resources(self, split):
        return {
            "metadata_file": f"{split}_metadata.jsonl"
        }


