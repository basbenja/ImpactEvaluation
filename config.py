import os

from dotenv import load_dotenv

_ = load_dotenv(override=True)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.getenv('DATA_DIR', os.path.join(THIS_DIR, 'data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)