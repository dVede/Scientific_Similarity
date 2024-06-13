import os
import fitz
import io
import numpy as np

from PIL import Image
from DocSim import get_layout_sim
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.segformer import load_model, load_processor
from surya.settings import settings

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    return None
        
def pdf_to_images(path):
    try:
        pdf_doc = fitz.open(path)
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return []
    
    images = []
    for page_num in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_num)
        pix = page.get_pixmap()
        image_bytes = pix.tobytes()
        image = Image.open(io.BytesIO(image_bytes))
        images.append(image)
    return images

def structural_similarity(draft, draft_path, corpus, corpus_path, layout_similarity_check = False):
    articles = [list(article) for article in corpus]
    draft_list = list(draft)
    for idx, article in enumerate(corpus):
        corpus_pdf_path = find(article[0] + '.pdf', corpus_path)
        if corpus_pdf_path:
            article.append(pdf_to_images(corpus_pdf_path))
        else:
            print(f"PDF for {article[0]} not found in {corpus_path}")
            article.append([])
    draft_pdf_path = find(draft[0] + '.pdf', draft_path)
    if draft_pdf_path:
        draft_list.append(pdf_to_images(draft_pdf_path))
    else:
        print(f"PDF for {draft[0]} not found in {draft_path}")
        draft_list.append([])
    count_similarity = count_sim(draft_list, articles)
    layout_sim = 0
    if (layout_similarity_check):
        layout_sim = layout_similarity(draft_list[8], [sublist[8] for sublist in articles])
    return count_similarity, layout_sim

def get_min_max_values(corpus, key):
    values = [item[key] for item in corpus]
    return min(values), max(values)

def elements_rules_check(draft, corpus):
    min_figures, max_figures = get_min_max_values(corpus, 'figures')
    min_tables, max_tables = get_min_max_values(corpus, 'tables')
    min_equations, max_equations = get_min_max_values(corpus, 'equations')
    figures_n = draft['figures']
    tables_n = draft['tables']
    equations_n = draft['equations']
    return (min_figures <= figures_n <= max_figures, 
            min_tables <= tables_n <= max_tables, 
            min_equations <= equations_n <= max_equations)

def is_size_in_interval(draft_len, corpus):
    min_len, max_len = min(corpus, key=len), max(corpus, key=len)
    return len(min_len) <= draft_len <= len(max_len)

def is_abstract_size_in_interval(draft, corpus):
    return is_size_in_interval(len(draft), [len(x.split()) for x in corpus])

def is_fulltext_size_in_interval(draft, corpus):
    draft_text_len = len(' '.join([sublist[1] for sublist in draft]).split(' '))
    corpus_lengths = [len(' '.join([sublist[1] for sublist in article]).split(' ')) for article in corpus]
    return is_size_in_interval(draft_text_len, corpus_lengths)

def is_in_interval(draft, corpus):
    return is_size_in_interval(len(draft), corpus)

def is_lesser(draft, corpus):
    return len(draft) <= max(len(x) for x in corpus)

def count_sim(draft, corpus):
    rules_list = [
        is_abstract_size_in_interval(draft[2], [sublist[2] for sublist in corpus]),
        is_in_interval(draft[3], [sublist[3] for sublist in corpus]),
        is_fulltext_size_in_interval(draft[3], [sublist[3] for sublist in corpus]),
        is_in_interval(draft[4], [sublist[4] for sublist in corpus]),
        *elements_rules_check(draft[5], [sublist[5] for sublist in corpus]),
        is_lesser(draft[6], [sublist[6] for sublist in corpus]),
        is_in_interval(draft[7], [sublist[7] for sublist in corpus]),
        is_in_interval(draft[8], [sublist[8] for sublist in corpus])
    ]
    similarity_count = rules_list.count(True) / len(rules_list)
    return similarity_count

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def layout_similarity(draft, corpus):
    model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    processor = load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    det_model = load_model()
    det_processor = load_processor()

    def predict_layouts(doc):
        layouts_predictions = []
        for group in chunker(doc, 16):
            line_predictions = batch_text_detection(group, det_model, det_processor)
            layouts_predictions.append(batch_layout_detection(group, model, processor, line_predictions))
        return layouts_predictions
    
    draft_layouts_predictions = predict_layouts(draft)
    corpus_layouts_predictions = [predict_layouts(article) for article in corpus]

    def extract_bboxes_labels(layouts_predictions):
        bboxes = [
            [bbox.bbox for bbox in layout_result.bboxes]
            for layouts_results in layouts_predictions
            for layout_result in layouts_results
        ]
        labels = [
            [bbox.label for bbox in layout_result.bboxes]
            for layouts_results in layouts_predictions
            for layout_result in layouts_results
        ]
        return bboxes, labels
    
    draft_layouts_bboxes, draft_layouts_labels = extract_bboxes_labels(draft_layouts_predictions)
    corpus_layouts_bboxes, corpus_layouts_labels = extract_bboxes_labels(
        [article for article_predictions in corpus_layouts_predictions for article in article_predictions]
    )

    pages_sim = []
    for draft_bboxes, draft_labels in zip(draft_layouts_bboxes, draft_layouts_labels):
        perfect_sim = get_layout_sim(draft_bboxes, draft_labels, draft_bboxes, draft_labels)
        max_page_sim = 0
        for corpus_bboxes, corpus_labels in zip(corpus_layouts_bboxes, corpus_layouts_labels):
            page_sim = get_layout_sim(draft_bboxes, draft_labels, corpus_bboxes, corpus_labels)
            max_page_sim = max(max_page_sim,page_sim)
        pages_sim.append(max_page_sim / perfect_sim)
    total_similarity = sum(pages_sim / len(pages_sim))
    return total_similarity