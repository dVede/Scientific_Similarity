import re
import bs4
from bs4 import BeautifulSoup
from typing import Dict
from dataclasses import dataclass

@dataclass
class Person:
    firstname: str
    middlename: str
    surname: str

SUBSTITUTE_TAGS = {
    'persName',
    'orgName',
    'publicationStmt',
    'titleStmt',
    'biblScope'
}

class TEIFile(object):
    def __init__(self, filename):
        self.filename = filename
        self.soup = parse_tei(filename)
        self._elements_amount = {}
        self._abstract = ''
        self._title = ''
        self._keywords = []
        self._authors = []
        self._text = []
        self._bib = []

    @property
    def key_words(self):
        if not self._keywords and self.soup.keywords:
            self._keywords = [kw.text for kw in self.soup.keyword.find_all("term")]
        return self._keywords

    @property
    def doi(self):
        idno_elem = self.soup.find('idno', type='DOI')
        return idno_elem.getText() if idno_elem else ''

    @property
    def title(self):
        if not self._title and self.soup.title:
            self._title = self.soup.title.getText()
        return self._title
    
    @property
    def elements_num(self):
        if not self._elements_amount:
            figures_amount = len([fig for fig in self.soup.body.find_all("figure") if not fig.has_attr("type")])
            tables_amount = len([fig for fig in self.soup.body.find_all("figure") if fig.has_attr("type")])
            equations_amount = len(self.soup.body.find_all("formula"))
            self._elements_amount = {
                'figures': figures_amount,
                'tables': tables_amount,
                'equations': equations_amount,
            }
        return self._elements_amount
    

    @property
    def abstract(self):
        if not self._abstract and self.soup.abstract:
            self._abstract = self.soup.abstract.getText(separator=' ', strip=True)
        return self._abstract

    @property
    def bib(self):
        if not self._bib and self.soup.listBibl:
            self._bib = [parse_bib_entry(entry) for entry in self.soup.listBibl.find_all("biblStruct") if parse_bib_entry(entry)['title']]
        return self._bib
    
    @property
    def authors(self):
        if not self._authors and self.soup.analytic:
            authors_in_header = self.soup.analytic.find_all('author')
            self._authors = [
                Person(
                    elem_to_text(author.persName.find("forename", type="first")),
                    elem_to_text(author.persName.find("middlename", type="middle")),
                    elem_to_text(author.persName.surname)
                )
                for author in authors_in_header if author.persName
            ]
        return self._authors

    @property
    def text(self):
        if not self._text:
            full_text = self.process_div(self.soup.body, "")
            combined_section = {}
            for text, section in full_text:
                deleted_cit_text = delete_citations(text)
                combined_section[section] = combined_section.get(section, '') + " " + deleted_cit_text
            self._text = [(text.strip(), section) for section, text in combined_section.items()]
        return self._text

    def process_div(self, div, sections):
            divs_text = []
            if div.div:
                for sub_div in div.find_all("div", recursive=False):
                    head_text = sub_div.head.text.strip()
                    new_sections = sections + head_text if head_text else sections
                    divs_text.extend(self.process_div(sub_div, new_sections))
            for tag in div:
                if tag.name == 'p' and tag.text:
                    divs_text.append((tag.text.strip(), sections))
            return divs_text

def get_venue_from_grobid_xml(raw_xml: BeautifulSoup, title_text: str) -> str:
    keep_types = ["j", "m", "s"]
    title_names = [
        (title_entry["level"], title_entry.text)
        for title_entry in raw_xml.find_all("title")
        if title_entry.get("level") in keep_types and title_entry.text != title_text
    ]
    if title_names:
        title_names.sort(key=lambda x: keep_types.index(x[0]))
        return title_names[0][1]
    return ""

def get_title_from_grobid_xml(raw_xml: BeautifulSoup) -> str:
    for title_entry in raw_xml.find_all("title"):
        if title_entry.get("level") == "a":
            return title_entry.text
    return raw_xml.title.text if raw_xml.title else ""

def clean_tags(el: bs4.element.Tag):
    for sub_tag in SUBSTITUTE_TAGS:
        for sub_el in el.find_all(sub_tag):
            sub_el.name = sub_tag.lower()

def parse_bib_entry(bib_entry: BeautifulSoup) -> Dict:
    clean_tags(bib_entry)
    title = get_title_from_grobid_xml(bib_entry)
    return {
        'ref_id': bib_entry.attrs.get("xml:id"),
        'title': title,
        'venue': get_venue_from_grobid_xml(bib_entry, title),
        'urls': []
    }

def elem_to_text(elem, default=''):
    return elem.getText() if elem else default

def parse_tei(tei_path):
    with open(tei_path, 'r', encoding="utf8") as tei:
        soup = BeautifulSoup(tei, 'xml')
        if soup:
            return soup
        else:
            raise RuntimeError('Cannot generate a soup from the input')

def delete_citations(text):
    patterns = [
        r'\[(\d+(\-\d+)?(,\s*\d+(\-\d+)?)*)\]',
        r'\((\d+(\-\d+)?(,\s*\d+(\-\d+)?)*)\)',
        r'\(.*?\d{4}[a-z]?\)|\(.*?\d{4}[a-z]?\)',
        r'\(?\b(tables?|tab\.)\s*\d[a-zA-Z]?\)?',
        r'\(?\b(equations?|eq\.)\s*\d[a-zA-Z]?\)?',
        r'\(?\b(figures?|fig\.)\s*\d[a-zA-Z]?\)'
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.I)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_fulltext(articles):
    sections_text_list = [
        delete_citations(text_segment)
        for article in articles
        for text_segment, _ in article[3]
    ]
    fulltext = ". ".join(sections_text_list)
    fulltext = delete_citations(fulltext)
    return fulltext.split(" ")
