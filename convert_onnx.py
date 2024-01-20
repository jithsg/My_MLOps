import torch
import hydra
import logging

from omegaconf.omegaconf import OmegaConf

from model import LitModel
from data import CIFAR10DataModule

logger = logging.getLogger(__name__)


@hydra.main(config_path="./configs", config_name="config")
def convert_model(cfg):
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/best-checkpoint.ckpt"
    logger.info(f"Loading pre-trained model from: {model_path}")
    lit_model = LitModel.load_from_checkpoint(model_path)

    data_model = CIFAR10DataModule(
        batch_size=cfg.datamodule.batch_size
    )
    data_model.prepare_data()
    data_model.setup()
    input_batch = next(iter(data_model.train_dataloader()))
    input_sample = {
        "input_ids": input_batch[0]
    }

    # Export the model
    logger.info(f"Converting the model into ONNX format")
    torch.onnx.export(
        lit_model, # model being run
        (
            input_sample["input_ids"],
 
        ),  # model input (or a tuple for multiple inputs)
        f"{root_dir}/models/model.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,
        opset_version=10,
        input_names=["input_ids"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input_ids": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    logger.info(
        f"Model converted successfully. ONNX format model is at: {root_dir}/models/model.onnx"
    )


if __name__ == "__main__":
    convert_model()