import weaviate
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.config_log as config_log

config, logger, CONFIG_PATH = config_log.setup_config_and_logging()
config.read(CONFIG_PATH)

def create_embedded_weaviate_client(openai_api_key, data_path):
    """
    Create and post back an Embedded Weaviate client.
    data_path will be used to store vector data and can be used the next time the program is started.
    """
    os.environ['OPENAI_APIKEY'] = config.get('OpenAI', 'api_key')
    os.environ['OPENAI_API_KEY'] = config.get('OpenAI', 'api_key')
    try:
        return weaviate.connect_to_embedded(
            hostname=config.get('Weaviate', 'host'),
            port=int(config.get('Weaviate', 'port')),
            grpc_port=int(config.get('Weaviate', 'grpc_port')),
            version="1.26.1",
            persistence_data_path=data_path,
            headers={
            "X-OpenAI-Api-Key": openai_api_key
            },
        )
    except:
        return weaviate.connect_to_local(
            host=config.get('Weaviate', 'host'),
            port=int(config.get('Weaviate', 'port')),
            grpc_port=int(config.get('Weaviate', 'grpc_port')),
        )
