from src.parser.web_page_parser import build_documents_from_urls
from src.script.persist_workflow import post_json

BASE_URL = "http://localhost:8000"
urls = ["https://www.ecfr.gov/current/title-40"]

documents = build_documents_from_urls(urls)
payload = {"documents": documents}
resp = post_json(f"{BASE_URL}/ingest", payload)
print(resp)