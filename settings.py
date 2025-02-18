from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

class Settings:
    FILES_FOLDER = os.getenv('FILES_FOLDER', './data')
    GROQ_KEY = os.getenv('GROQ_KEY')
    GROQ_MODEL = os.getenv('GROQ_MODEL')
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    FAST_EMBDED_MODEL = os.getenv("FAST_EMBDED_MODEL")
    SQL_SERVER_HOST = os.getenv('SQL_SERVER_HOST')
    SQL_SERVER_DATABASE = os.getenv('SQL_SERVER_DATABASE')
    SQL_SERVER_USERNAME = os.getenv('SQL_SERVER_USERNAME')
    SQL_SERVER_PASSWORD = os.getenv('SQL_SERVER_PASSWORD')
    LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
    OPENAI_EMBEDDINGS_MODEL = os.getenv('OPENAI_EMBEDDINGS_MODEL')

    @classmethod
    def check(cls):
        required =   'GROQ_KEY', 'GROQ_MODEL', 'OPENAI_API_KEY'
        for var in required:
            if not getattr(cls, var):
                raise ValueError(f"Missing {var} in environment")
            
# Instantiate the settings class and check required variables
settings = Settings()
settings.check()


# print(settings.GROQ_MODEL)
# print(settings.GROQ_KEY)
# Access the variables through the settings object
# print(settings.SQL_SERVER_DATABASE)  # This will now correctly print the value
