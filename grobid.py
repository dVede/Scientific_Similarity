from grobid_client.grobid_client import GrobidClient
from grobid_client.grobid_client import ServerUnavailableException

def generate_tei_files(draft, corpus):
    try:
        client = GrobidClient(config_path="./config.json")
        client.process(output = draft + "/tei", service="processFulltextDocument", tei_coordinates=False, input_path=draft, n=10)
        client.process(output = corpus + "/tei", service="processFulltextDocument", tei_coordinates=True, input_path=corpus, n=20)
    except ServerUnavailableException:
        return False
    return True