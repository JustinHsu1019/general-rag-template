import time
import uuid
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import utils.config_log as config_log
from utils.embedded_weaviate_service import create_embedded_weaviate_client
from langchain_text_splitters import RecursiveCharacterTextSplitter

import weaviate
from weaviate.classes.config import Configure, Property, DataType


# Read the configuration file and initialize the logger
config, logger, CONFIG_PATH = config_log.setup_config_and_logging()
config.read(CONFIG_PATH)

# Read parameters from configuration file
PERSIST_PATH = config.get('Weaviate', 'persistence_data_path', fallback='./my_embedded_db')
openai_api_key = config.get('OpenAI', 'api_key')


class WeaviateManager:
    def __init__(self, collection_name):
        self.client = create_embedded_weaviate_client(
            openai_api_key=openai_api_key,
            data_path=PERSIST_PATH
        )
        self.collection_name = collection_name
        self.check_collection_exists()

    def check_collection_exists(self):
        """
        Check if collection already exists, if not create it.
        In v4, schema is no longer used, but collections.create(...) is used
        """
        if self.client.collections.exists(self.collection_name):
            print(f'{self.collection_name} is ready')
            return True

        print(f'Creating collection: {self.collection_name} ...')
        self.client.collections.create(
            name=self.collection_name,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(
                model="text-embedding-3-large",
            ),
            properties=[
                Property(name="uuid", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
            ]
        )
        print(f'{self.collection_name} is ready')
        return True

    def insert_data(self, content):
        """
        Add a new data object to Weaviate.
        Correct usage of v4: collection = client.collections.get(...)
                    collection.data.insert(...)
        """
        collection = self.client.collections.get(self.collection_name)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                new_uuid = str(uuid.uuid4())
                # Directly use insert() to add
                inserted_id = collection.data.insert({
                    "uuid": new_uuid,
                    "content": content
                })
                break
            except weaviate.exceptions.UnexpectedStatusCodeException as e:
                # If there is a 429 (rate limit) situation, consider retry
                if "429" in str(e):
                    print(f'Rate limit exceeded, retrying in 5 seconds... (Attempt {attempt + 1}/{max_retries})')
                    time.sleep(5)
                else:
                    raise


if __name__ == '__main__':
    """
    Sample main program: read the file, divide the text into chunks, and insert them into Weaviate one by one
    """
    manager = WeaviateManager(config.get('Weaviate', 'class_name'))

    with open('data/File_6328.txt', encoding='utf-8') as file:
        content = file.read()

    # Use langchain's TextSplitter to make chunk + 500 token overlap based on 2000 tokens
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    datas = text_splitter.split_text(content)

    for i, chunk_text in enumerate(datas, start=1):
        manager.insert_data(chunk_text)
        print(f"Successful Insert {i}th: {chunk_text[:10]} ...")
