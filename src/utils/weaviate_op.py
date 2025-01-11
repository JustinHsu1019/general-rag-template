import os
import sys

import voyageai
from langchain.embeddings import OpenAIEmbeddings
from weaviate.classes.query import MetadataQuery

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.config_log as config_log
from utils.embedded_weaviate_service import create_embedded_weaviate_client

config, logger, CONFIG_PATH = config_log.setup_config_and_logging()
config.read(CONFIG_PATH)

VOYAGE_APIKEY = config.get('VoyageAI', 'api_key')
WEA_CLASSNM = config.get('Weaviate', 'class_name')
PROPERTIES = ['uuid', 'content']

os.environ['OPENAI_APIKEY'] = config.get('OpenAI', 'api_key')
os.environ['OPENAI_API_KEY'] = config.get('OpenAI', 'api_key')
PERSIST_PATH = config.get('Weaviate', 'persistence_data_path', fallback='./my_embedded_db')


def rerank_with_voyage(query, documents, api_key, topk):
    """
    Reorder incoming documents using VoyageAI’s rerank feature
    documents are [{'uuid':..., 'content':...}, ...]
    """
    vo = voyageai.Client(api_key=api_key)
    content_list = [doc['content'] for doc in documents]
    reranking = vo.rerank(query, content_list, model='rerank-2', top_k=topk)

    # Get the reordered indices and map them back to the original documents
    sorted_indices = [result.index for result in reranking.results]
    top_documents = [documents[i] for i in sorted_indices]

    return top_documents


class WeaviateSemanticSearch:
    def __init__(self, classnm):
        """
        classnm 即 Collection 名稱 (v3 用 Class，v4 改名為 Collection)
        """
        self.embeddings = OpenAIEmbeddings(chunk_size=1, model='text-embedding-3-large')
        self.client = create_embedded_weaviate_client(
            openai_api_key=os.environ['OPENAI_API_KEY'],
            data_path=PERSIST_PATH
        )
        self.classnm = classnm

    def aggregate_count(self):
        """
        Use Raw GraphQL query to obtain the total number of objects in the collection (count).
        The old query.aggregate(...).with_meta_count() method is not supported in v4.
        """
        gql_agg = f"""
        {{
          Aggregate {{
            {self.classnm} {{
              meta {{
                count
              }}
            }}
          }}
        }}
        """
        result = self.client.query.raw(gql_agg)
        # If you want to return only the count value, you can parse it yourself:
        # count_val = result['data']['Aggregate'][self.classnm][0]['meta']['count']
        return result

    def get_all_data(self, limit=100):
        """
        To obtain the data in this collection, you can specify limit.
        v4: First confirm that the collection exists, and then retrieve the object through collection.query.get(...).
        """
        if self.client.collections.exists(self.classnm):
            coll = self.client.collections.get(self.classnm)
            # Get the properties specified by PROPERTIES
            response = coll.query.get(attributes=PROPERTIES, limit=limit)
            # The returned response.objects is a list, each element has
            # .id (Weaviate internal UUID)
            # .properties (actually stored fields, such as uuid, content, etc.)
            return response.objects
        else:
            raise Exception(f'Collection {self.classnm} does not exist.')

    def delete_class(self):
        """
        Delete the collection (use schema.delete_class for v3, use collections.delete for v4)
        """
        if self.client.collections.exists(self.classnm):
            self.client.collections.delete(self.classnm)
        else:
            print(f"Collection {self.classnm} does not exist, skip delete.")

    def hybrid_search(self, query, num, alpha):
        """
        v4 Hybrid Search:
          -alpha=1: pure vector search
          -alpha=0: Pure keyword search (BM25F)
        The return contains a customized list of objects, including 'uuid', 'content', and score.
        """

        coll = self.client.collections.get(self.classnm)
        # Bring metadata.score /metadata.distance reference
        response = coll.query.hybrid(
            query=query,
            alpha=alpha,
            limit=num,
            return_metadata=MetadataQuery(score=True, distance=True)
        )

        # Encapsulate the query results and send them back
        results = []
        for obj in response.objects:
            # Get the uuid and content defined by user
            # Note: obj.properties.get('uuid') here is different from obj.id
            results.append({
                'uuid': obj.properties.get('uuid'),
                'content': obj.properties.get('content'),
                '_additional': {
                    'distance': obj.metadata.distance,
                    'score': obj.metadata.score
                }
            })
        return results


def search_do(input_):
    """
    Combined with Hybrid Search + Rerank.
    First use Hybrid to find 100 transactions, and then use VoyageAI Rerank to select the top 5.
    """
    HYBRID_SEARCH_NUM = 100
    RERANKER_NUM = 5

    # 80% Vector Search, 20% Keyword Search (personal experience concludes that 8:2 is the universal optimal solution)
    alp = 0.8

    searcher = WeaviateSemanticSearch(WEA_CLASSNM)

    # 1. Hybrid Search captures 100 transactions first
    results = searcher.hybrid_search(input_, HYBRID_SEARCH_NUM, alpha=alp)

    # 2. Organize into documents = [{'content':..., 'uuid':...}, ...] for rerank
    documents = [{'content': r['content'], 'uuid': r['uuid']} for r in results]

    # 3. 利用 VoyageAI rerank，再挑 5 筆
    reranked_documents = rerank_with_voyage(input_, documents, VOYAGE_APIKEY, RERANKER_NUM)

    # 只取 content 回傳
    result_contents = [doc['content'] for doc in reranked_documents]
    return result_contents


if __name__ == '__main__':
    vdb = WEA_CLASSNM
    client = WeaviateSemanticSearch(vdb)

    # Count the number of transactions
    count_result = client.aggregate_count()
    print(count_result)
    # If you only want count:
    # count_val = count_result['data']['Aggregate'][vdb][0]['meta']['count']
    # print(count_val)

    # Get all information
    # all_data = client.get_all_data()
    # print(all_data)

    # Delete collection
    # client.delete_class()

    # Test search_do()
    # quest = "Hello"
    # print(search_do(quest, alp=0.8))
