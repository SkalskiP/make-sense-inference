import os.path
from glob import glob


def main() -> None:
    serving_logic_directory = os.path.abspath(os.environ["SERVING_LOGIC_DIR"])
    serving_modules_paths = glob(os.path.join(serving_logic_directory, "*.py")) + glob(
        os.path.join(serving_logic_directory, "**", "*.py")
    )
    model_files = glob(os.path.join(os.path.abspath(os.environ["MODEL_DIR"]), "*"))
    print(",".join(serving_modules_paths + model_files))


if __name__ == "__main__":
    main()
