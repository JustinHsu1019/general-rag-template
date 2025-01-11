import json

import requests

import utils.config_log as config_log

config, logger, CONFIG_PATH = config_log.setup_config_and_logging()
config.read(CONFIG_PATH)


def gemini_template(prompt):
    api_key = config.get('Gemini', 'api_key')
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}'
    payload = {'contents': [{'parts': [{'text': prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    response = (requests.post(url, headers=headers, data=json.dumps(payload))).json()
    return response['candidates'][0]['content']['parts'][0]['text']


if __name__ == '__main__':
    prompt = """Tell me 3 tips for CTF reverse analysis, output in json format: {"Trick 1": ,"Trick 2": ,"Trick 3": }"""
    response = gemini_template(prompt)
    print(response)
