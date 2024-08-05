import os
from dotenv import load_dotenv

load_dotenv() # Load the environment variables

OPEN_AI_KEY = os.getenv('OPENAI_API_KEY')