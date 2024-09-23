import argparse

from transformers import AutoModel


def main(arg):
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
