from pipelines.train import finetune_pipeline
from utils.logger import logger

if __name__ == "__main__":

    logger.info("Starting the Finetuning Pipeline for google/flan-t5-small")
    finetune_pipeline()
