import os

import argparse
import sys
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer
from scope_check import scope_check
from grobid import generate_tei_files
from cocitation import cocitation_sim
from semantic_similarity import semantic_similarity
from TEIFile import TEIFile
from multiprocessing.pool import Pool
from structural_similarity import structural_similarity

from pathlib import Path
from os.path import basename, splitext

def basename_without_ext(path):
    base_name = basename(path)
    stem, ext = splitext(base_name)
    if stem.endswith('.grobid.tei'):
        return stem[0:-11]
    else:
        return stem
    
def tei_to_csv_entry(tei_file):
    tei = TEIFile(tei_file)
    base_name = basename_without_ext(tei_file)
    return base_name, tei.title, tei.abstract, tei.text, tei.bib, tei.elements_num, tei.authors, tei.key_words

def similarity_calculation(draft, draft_path, corpus, corpus_path, is_layout):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoAdapterModel.from_pretrained('allenai/specter2_aug2023refresh_base')
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_aug2023refresh_base')
    model.load_adapter("allenai/specter2_aug2023refresh_classification", source="hf", load_as="specter2_classification", set_active=True)
    model.to(device)
    is_in_scope = scope_check(draft, corpus, model, tokenizer, device)
    cocitation_most_similar, cocitation_similarity = cocitation_sim(draft, corpus)
    semantic_most_similar1, semantic_most_similar2, semantic_sim = semantic_similarity(draft, corpus, model, tokenizer, device)
    structural_similarity1, structural_similarity2 = structural_similarity(draft, draft_path, corpus, corpus_path, is_layout)
    multiplier = 25
    if is_layout:
        multiplier = 20
    sum_sim = (cocitation_similarity + semantic_sim + structural_similarity1 + structural_similarity2 + int(is_in_scope)) * multiplier
    print(f'============================================')
    print(f'Is paper in scope? --- {is_in_scope}')
    print(f'============================================')
    print(f'Most similar semantic paper (title + abstract) --- {structural_similarity1}')
    print(f'Most similar semantic paper (full text) --- {structural_similarity2}')
    print(f'Semanric similarity = {semantic_sim * 100}%')
    print(f'============================================')
    print(f'Most similar citation paper --- {cocitation_most_similar}')
    print(f'Citation similarity = {cocitation_similarity * 100}%')
    print(f'============================================')
    print(f'Compliance with manuscript template requirements = {structural_similarity1 * 100}%')
    if is_layout:
        print(f'Layout similarity = {structural_similarity2 * 100}%')
    print(f'============================================')
    print(f'Similarity between {draft[0]} and corpora = {sum_sim}')
    return sum_sim

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process draft and pirectory with journals")
    
    parser.add_argument('-d', '--draft', type=str, required=True, help='Path to the draft directory')
    parser.add_argument('-j', '--journal', type=str, required=True, help='Path to the journal articles directory')
    parser.add_argument('-l', '--layout', type=bool, required=False, help='Do layout docsim analysis')

    args = parser.parse_args()

    draft_path = args.draft
    corpus_path = args.journal
    is_layout = args.layout

    if not os.path.isdir(draft_path):
        print(f"Error: the draft path '{draft_path}' does not exist or is not a directory")
        sys.exit(1)

    if not os.path.isdir(corpus_path):
        print(f"Error: the journal articles path '{corpus_path}' does not exist or is not a directory")
        sys.exit(1)

    is_generated = generate_tei_files(draft_path, corpus_path)
    if not is_generated:
        print(f"Start grobid container before starting evaluate similarity")
        sys.exit(1)

    corpus_papers = sorted(Path(corpus_path + "/tei").glob('*.tei.xml'))
    pool = Pool()    
    corpus = pool.map(tei_to_csv_entry, corpus_papers)
    pool.close()
    pool.join()
    draft = tei_to_csv_entry(draft_path + "/tei")
    
    similarity_calculation(draft, draft_path, corpus, corpus_path, is_layout)



