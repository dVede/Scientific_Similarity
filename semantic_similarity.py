import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def semantic_similarity(draft, corpus, model, tokenizer, device):
    most_similar_paper1, semantic_similarity1 = header_semantic_similarity(draft, corpus, model, tokenizer, device)
    most_similar_paper2, semantic_similarity2 = fulltext_semantic_similarity(draft, corpus, model, tokenizer, device)
    return most_similar_paper1, most_similar_paper2, (semantic_similarity2 + semantic_similarity1) / 2

def header_semantic_similarity(draft, corpus, model, tokenizer, device):
    max_semantic = 0.0
    most_similar_paper = ''
    draft_embedding = get_specter2_embedding_header(draft[1], draft[2], model, tokenizer, device)
    for paper in corpus:
        paper_embedding = get_specter2_embedding_header(paper[1], paper[2], model, tokenizer, device)
        similarity = get_embeddings_similarity(draft_embedding, paper_embedding)
        if max_semantic < similarity:
            max_semantic = similarity
            most_similar_paper = paper[1]
    return most_similar_paper, max_semantic

def fulltext_semantic_similarity(draft, corpus, model, tokenizer, device):
    max_semantic = 0.0
    most_similar_paper = ''
    draft_embedding = get_specter2_embedding_text(draft[3], model, tokenizer, device)
    for paper in corpus:
        paper_embedding = get_specter2_embedding_text(paper[3], model, tokenizer, device)
        similarity = get_embeddings_similarity(draft_embedding, paper_embedding)
        if max_semantic < similarity:
            max_semantic = similarity
            most_similar_paper = paper[1]
    return most_similar_paper, max_semantic

def get_embeddings_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(torch.tensor(embedding1),torch.tensor(embedding2)).item()

def get_specter2_embedding_keywords(keywords, model, tokenizer, device):
    inputs = tokenizer(keywords, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def get_specter2_embedding_header(title, abstract, model, tokenizer, device):
    inputs = tokenizer(tokenizer.cls_token + title + tokenizer.sep_token + abstract, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def get_specter2_embedding_text(sections, model, tokenizer, device):
    sections_embeddings = []
    for text_segment, section in sections:
        splitted_texts = split_text_into_subsections(text_segment, tokenizer)
        splitted_texts[0] = section + ". " + splitted_texts[0]
        subsection_embeddings = []
        for splitted_text in splitted_texts:
            marked_text = tokenizer.cls_token + splitted_text
            inputs = tokenizer(marked_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            subsection_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
        mean_section_embedding = np.mean(subsection_embeddings, axis = 0)
        sections_embeddings.append(mean_section_embedding)
    mean_text_embedding = np.mean(sections_embeddings, axis = 0)
    return mean_text_embedding

def split_text_into_subsections(text, tokenizer, max_tokens=512):
    sentences = text.split(". ")
    subsections = []
    current_subsection = ""
    current_length = 0
    for sentence in sentences:
        sentence_length = len(tokenizer.tokenize(sentence))
        if current_length + sentence_length > max_tokens:
            subsections.append(current_subsection)
            current_subsection = sentence + ". "
            current_length = sentence_length
        else:
            current_subsection += sentence + ". "
            current_length += sentence_length
    if current_subsection:
        subsections.append(current_subsection.strip())

    return subsections