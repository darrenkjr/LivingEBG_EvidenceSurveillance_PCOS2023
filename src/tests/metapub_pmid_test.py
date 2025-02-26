
import dotenv
dotenv.load_dotenv()
from metapub import PubMedFetcher

fetch = PubMedFetcher()
article = fetch.article_by_pmid(22132633)
print(article)

