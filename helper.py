# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv


# these expect to find a .env file at the directory above the lesson.    
# the format for that file is (without the comment)                      
# API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService                                                            
def load_env():
    _ = load_dotenv(find_dotenv())


def get_groq_api_key():
    """Get Groq API key from environment variables"""
    load_env()
    groq_api_key = os.getenv("GROQ_API_KEY")
    return groq_api_key


def get_openai_api_key():
    """Legacy function - kept for compatibility with other lessons"""
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key


def get_mlflow_tracking_uri():
    return "http://localhost:8080"
    #return os.environ.get('DLAI_LOCAL_URL').format(port=8080)
