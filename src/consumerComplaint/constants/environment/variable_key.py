import os 
from dotenv import load_dotenv
load_dotenv() 

AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
MONGO_DB_URL_ENV_KEY = os.getenv("MONGO_DB_URL")

