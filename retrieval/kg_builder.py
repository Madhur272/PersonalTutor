import spacy
import networkx as nx
import json
from collections import defaultdict
import re

# Use a more powerful model that includes dependency parsing
nlp = spacy.load("en_core_web_sm")

# --- Constants ---
CHUNKS_PATH = "D:/college/sem 7/CL&NLP/PersonalTutor/Project/extractor/chunks_class_9_English_Beehive.json"
KG_EXPORT_PATH = "kg_export.json"

# --- Main Functions ---

def clean_entity_text(text):
    """Normalize entity text: strip, lowercase, remove leading articles.

    This helps deduplicate nodes that differ only by case or leading articles.
    """
    text = text.strip()
    # remove leading articles
    text = re.sub(r'^(the|a|an)\s+', '', text, flags=re.IGNORECASE)
    # normalize whitespace and lower-case for canonical node ids
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def compound_phrase(token):
    """Return a compound phrase for a token by joining left-side compound children.

    e.g. for token 'City' with left compound children 'New' and 'York' -> 'New York City'
    This preserves the surface form order and is used to build more informative node ids
    when spaCy NER does not capture the full compound.
    """
    parts = []
    # left-side compound modifiers typically appear as children with dep_ == 'compound'
    for child in token.lefts:
        if child.dep_ == 'compound':
            parts.append(child.text)
    parts.append(token.text)
    return ' '.join(parts)

def extract_svo_relations(doc, chunk_metadata):
    """
    Extracts Subject-Verb-Object triplets and entity relations using dependency parsing.
    This is a significant improvement over the keyword-based rule.
    """
    relations = []
    
    for token in doc:
        # VERB-based relations (subject -> object)
        if token.pos_ == 'VERB':
            subjects = []
            for child in token.children:
                if child.dep_ in ('nsubj', 'nsubjpass'):
                    # prefer compound-aware phrase for nouns
                    if child.pos_ in ('NOUN', 'PROPN'):
                        subj_phrase = compound_phrase(child)
                        subjects.append(clean_entity_text(subj_phrase))
                    else:
                        subjects.append(clean_entity_text(child.text))

            # direct objects
            objects = []
            for child in token.children:
                if child.dep_ == 'dobj':
                    if child.pos_ in ('NOUN', 'PROPN'):
                        obj_phrase = compound_phrase(child)
                        objects.append(clean_entity_text(obj_phrase))
                    else:
                        objects.append(clean_entity_text(child.text))

            # prepositional objects: verb -> prep -> pobj (e.g., 'talk about climate')
            for prep in [c for c in token.children if c.dep_ == 'prep']:
                for pobj in [c2 for c2 in prep.children if c2.dep_ == 'pobj']:
                    if pobj.pos_ in ('NOUN', 'PROPN'):
                        objects.append(clean_entity_text(compound_phrase(pobj)))
                    else:
                        objects.append(clean_entity_text(pobj.text))

            if subjects and objects:
                for subj in subjects:
                    for obj in objects:
                        relation_data = {
                            "relation": token.lemma_.lower(),
                            **chunk_metadata
                        }
                        relations.append((subj, obj, relation_data))

        # ATTR/COPA relations (copula constructions, attributes)
        if token.dep_ in ("attr", "acomp"):
            # head (often a nominal) relates to the attribute token
            head = token.head
            if head.pos_ in ('NOUN', 'PROPN'):
                subj = clean_entity_text(compound_phrase(head))
            else:
                subj = clean_entity_text(head.text)
            if token.pos_ in ('NOUN', 'PROPN'):
                obj = clean_entity_text(compound_phrase(token))
            else:
                obj = clean_entity_text(token.text)
            relation_data = {"relation": token.dep_, **chunk_metadata}
            relations.append((subj, obj, relation_data))

        # Compound handling: when a noun has compound children, create a combined token
        # This is handled at node-creation time by using spaCy entity spans where possible.
    return relations

def build_knowledge_graph(chunks_path):
    """
    Main function to build and return the knowledge graph from text chunks.
    """
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {chunks_path} was not found.")
        return None

    G = nx.DiGraph()
    
    print(f"Processing {len(chunks)} chunks...")
    for chunk in chunks:
        doc = nlp(chunk["text"])
        
        # 1. Add entities as nodes (normalize to lowercase canonical form)
        for ent in doc.ents:
            entity_text = clean_entity_text(ent.text)
            # Use canonicalized node id; store original surface forms as 'surface' list
            if not G.has_node(entity_text):
                G.add_node(entity_text, label=ent.label_, surfaces=[ent.text])
            else:
                # update label if necessary and append surface form if new
                G.nodes[entity_text]['label'] = ent.label_
                if ent.text not in G.nodes[entity_text].get('surfaces', []):
                    G.nodes[entity_text].setdefault('surfaces', []).append(ent.text)

        # 2. Extract and add relations as edges
        chunk_metadata = {
            "chunk_id": chunk.get("chunk_id"),
            "page_number": chunk.get("page_number"),
            "chapter": chunk.get("chapter"),
            "book": chunk.get("book")
        }
        relations = extract_svo_relations(doc, chunk_metadata)

        for src, tgt, attrs in relations:
            # Ensure both nodes exist before adding an edge
            if not G.has_node(src):
                G.add_node(src, label="UNKNOWN", surfaces=[src])
            if not G.has_node(tgt):
                G.add_node(tgt, label="UNKNOWN", surfaces=[tgt])

            # Also attach the chunk text to the edge attributes (useful later)
            edge_attrs = dict(attrs)
            edge_attrs.setdefault('chunk_text', chunk.get('text', ''))
            G.add_edge(src, tgt, **edge_attrs)

    return G

def save_graph(graph, output_path):
    """Saves the NetworkX graph to a JSON file."""
    if graph is None:
        print("Graph is empty. Nothing to save.")
        return
        
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    data = nx.readwrite.json_graph.node_link_data(graph)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Knowledge graph saved to {output_path}")


# --- Execution ---
if __name__ == "__main__":
    knowledge_graph = build_knowledge_graph(CHUNKS_PATH)
    if knowledge_graph:
        save_graph(knowledge_graph, KG_EXPORT_PATH)