from bs4 import BeautifulSoup
import html

def clean_html_tags(text):

    text = html.unescape(text)

    soup = BeautifulSoup(text, 'html.parser')
    clean_text = soup.get_text(separator = ' ' ,strip = True)
    clean_text = ' '.join(clean_text.split())
    return clean_text