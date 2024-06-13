from collections import Counter

def cocitation_sim(draft, corpus):
    draft_bib_titles = set(sublist['title'].lower() for sublist in draft[4])
    all_bib_titles = Counter(draft_bib_titles)
    for paper in corpus:
        paper_bib_titles = set(sublist['title'].lower() for sublist in paper[4])
        all_bib_titles.update(paper_bib_titles)
    max_cosim = 0.0
    for paper in corpus:
        paper_bib_titles = set(sublist['title'].lower() for sublist in paper[4])
        intersection = draft_bib_titles & paper_bib_titles
        inter_sum = sum(all_bib_titles[title] for title in intersection)
        unique_sum = sum(all_bib_titles[title] for title in draft_bib_titles | paper_bib_titles)
        cosim = inter_sum / unique_sum
        max_cosim = max(max_cosim, cosim)
    return max_cosim