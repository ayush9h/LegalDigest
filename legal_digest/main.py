from pipelines.inference import inference
from pipelines.train import finetune_pipeline
from utils.logger import logger

if __name__ == "__main__":

    # logger.info("Starting the Finetuning Pipeline for google/flan-t5-small")
    # finetune_pipeline()

    # logger.info("Inference model")
    user_query = input("Enter your query: ")
    answer = inference("results/checkpoint-383", user_query=user_query)
    logger.info(f"got the answer:{answer}")
