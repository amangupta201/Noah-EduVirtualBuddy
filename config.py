import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/ec2-user/.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
