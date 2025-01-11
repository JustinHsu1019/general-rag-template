import utils.config_log as config_log
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from flask_restx import Api, Resource, fields
from utils.ai.call_ai import call_llm
from utils.weaviate_op import search_do
from utils.weaviate_op import WeaviateSemanticSearch
from werkzeug.security import check_password_hash, generate_password_hash

config, logger, CONFIG_PATH = config_log.setup_config_and_logging()
config.read(CONFIG_PATH)

auth = HTTPBasicAuth()

users = {'rag': generate_password_hash(config.get('Api_docs', 'password'))}

WEAV_ENG = WeaviateSemanticSearch(config.get('Weaviate', 'class_name'))

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username
    return None


app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type, Qs-PageCode, Cache-Control'

api = Api(
    app,
    version='1.0',
    title='rag API',
    description='rag API'
)

ns = api.namespace('api', description='Chatbot operations')

model = api.model(
    'ChatRequest',
    {
        'message': fields.String(required=True, description='User input message.')
    },
)


@ns.route('/')
class HealthCheck(Resource):
    @api.doc('health_check')
    def get(self):
        """Server health check."""
        response = jsonify('server is ready')
        response.status_code = 200
        return response


@ns.route('/chat')
class ChatBot(Resource):
    @api.doc('chat_bot')
    @api.expect(model)
    def post(self):
        question = request.json.get('message')

        use_gpt = True

        if not question:
            response = jsonify({'llm': 'no question', 'retriv': []})
            response.status_code = 200
            return response
        else:
            try:
                r_contents = search_do(question)
                print(r_contents)

                content_list = ""
                for i in range(len(r_contents)):
                    content_list = content_list + "references" + str(i+1) + ":\n" + r_contents[i] + "\n\n"

                response = call_llm(content_list, question, use_gpt)

                if not isinstance(response, str):
                    response = str(response)

            except Exception:
                response = jsonify({'message': 'Internal Server Error'})
                response.status_code = 500
                return response

        try:
            # "llm" is the response message from the LLM (String); "retriv" is all the data referenced by the LLM in its response (List).
            response = jsonify({'llm': response, 'retriv': r_contents})
            response.status_code = 200
            return response
        except TypeError:
            response = jsonify({'message': 'Internal Server Error'})
            response.status_code = 500
            return response


@app.before_request
def require_auth_for_docs():
    if request.path == '/':
        return auth.login_required()(swagger_ui)()


@app.route('/')
def swagger_ui():
    return api.render_doc()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
