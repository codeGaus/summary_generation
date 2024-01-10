import subprocess
import json


def read_model_response():
    with open('data/tmp/last_json.txt') as f:
        data = f.read()
    return json.loads(data)


def create_xml(data):
    with open('data/diagram/data.xml', 'w') as f:
        f.write(data['Theme'])
        for i, q in enumerate(data['MainQuestions']):
            f.write('  ' + q['Question'] + '\n')
            if data['SecondaryQuestions'][i].get('Question', None):
                f.write('    ' + data['SecondaryQuestions'][i]['Question'] + '\n')


subprocess.run(["diomindmap generate", "-i data/diagram/data.xml", "-o graph.drawio"])
