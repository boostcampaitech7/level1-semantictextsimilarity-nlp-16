import argparse

from transformers import AutoModel


def main(arg):
    """_summary_
    LoRA 적용을 위한 module name print

    특정 모델에 대해 LoRA 적용을 위해,
    해당 모델을 로드하여 모델의 module name을 print
    Args:
        arg (argparse.Namespace): command line arguments
            - model_name : module name을 검색할 model
    """
    MODEL_NAME = arg.model_name
    model = AutoModel.from_pretrained(MODEL_NAME)
    for name, module in model.named_modules():
        print(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        help="Enter the name of a model that requires module names for the mode",
    )
    arg = parser.parse_args()

    main(arg)
