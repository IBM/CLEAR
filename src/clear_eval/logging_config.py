import logging

def setup_logging(level=logging.INFO):
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(lineno)d -  %(levelname)s - %(message)s",
            level=logging.INFO,
        )
    for noisy_lib in ["requests", "urllib3", "openai", "ibm_watsonx_ai", "httpx"]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)