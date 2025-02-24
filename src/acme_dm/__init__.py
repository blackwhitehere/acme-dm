import logging

from dotenv import load_dotenv

from acme_dm.data_metrics import add_new_metrics, create_metrics  # noqa

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(pathname)s | %(name)s | func: %(funcName)s:%(lineno)s | %(levelname)s | %(message)s",
)
