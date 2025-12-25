import pdfplumber
import pandas as pd
import re
import nltk
import json
from collections import OrderedDict

def sentence_tokenize(text):
    try:
        return nltk.sent_tokenize(text)
    except (LookupError, Exception):
        return [p.strip() for p in re.split(r'(?<=[\.\?\!])\s+', text) if p.strip()]

def reduce_repeated_characters(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)

def extract_and_chunk_chapter(pdf_path: str, start_page: int, end_page: int, chapter: str, chunk_size: int = 2) -> list[dict]:
    exercise_patterns = [
        r'^\s*Thinking about the Text\b',
        r'^\s*Thinking about the Poem\b',
        r'^\s*Thinking about Language\b',
        r'^\s*Thinking about the Language\b',
        r'^\s*Answer these questions\b',
        r'^\s*Working with the Text\b',
        r'^\s*Working with Language\b',
        r'^\s*Speaking\b',
        r'^\s*Writing\b',
        r'^\s*Project\b',
        r'^\s*Do a Project\b',
        r'^\s*Do you agree\b',
        r'^\s*Complete the following\b',
        r'^\s*Exercises\b'
    ]

    teacher_note_markers = [
        "Let the children talk freely",
        "You might want to explain",
        "In this unit students are required",
        "The following points could be explained"
    ]

    sentences_with_pages = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        if start_page < 1 or start_page > total:
            raise ValueError("start_page out of range")
        if end_page is None:
            end_page = total
        end_page = min(end_page, total)
        for p in range(start_page, end_page + 1):
            raw = pdf.pages[p-1].extract_text(x_tolerance=2) or ""
            raw = reduce_repeated_characters(raw)
            if any(marker.lower() in raw.lower() for marker in teacher_note_markers):
                continue
            raw = re.sub(r'\s+', ' ', raw).strip()  

            cut_pos = None
            for pat in exercise_patterns:
                m = re.search(pat, raw, flags=re.IGNORECASE)
                if m:
                    cut_pos = m.start()
                    break
            if cut_pos is not None:
                raw = raw[:cut_pos]
            if not raw:
                continue

            sents = sentence_tokenize(raw)
            for s in sents:
                if len(s.split()) > 2:
                    sentences_with_pages.append((s.strip(), p))

    chunks = []
    for i in range(0, len(sentences_with_pages), chunk_size):
        group = sentences_with_pages[i:i+chunk_size]
        if not group:
            continue
        first_page = group[0][1]
        text = " ".join(g[0] for g in group)
        chunks.append({
            "chapter": chapter,
            "chunk_id": i // chunk_size + 1,
            "page_number": first_page,
            "text": text
        })

    return chunks


def generate_book_config_from_pdf(pdf_path: str, book_name: str = "Beehive", toc_max_page: int = 10, start_page_correction: int = 2) -> dict:
    chapter_regex = re.compile(r'^(\d{1,2})\.\s+(.*?)\s+\.{3,}\s+(\d+)$')
    chapter_starts = OrderedDict()
    logical_page_offset = None

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for i in range(min(total_pages, 20)): 
            text = pdf.pages[i].extract_text() or ""
            if re.search(r'\bNotes for the Teacher\b', text, re.IGNORECASE):
                logical_page_offset = i
                print(f"✅ Found 'Notes for the Teacher' (Logical Page 1) at PDF Page {i+1}")
                break

        if logical_page_offset is None:
            raise RuntimeError("❌ Could not locate 'Notes for the Teacher' to determine logical page offset.")

        for i in range(6, toc_max_page + 1):
            text = pdf.pages[i - 1].extract_text() or ""
            print(f"\n📄 Page {i}")
            for line in text.splitlines():
                print("  Line:", line.strip())
                match = chapter_regex.match(line.strip())
                if match:   
                    chap_num, title, page_str = match.groups()
                    try:
                        logical_page = int(page_str)
                        actual_pdf_page = logical_page + logical_page_offset
                        chapter_starts[title.strip()] = actual_pdf_page
                        print(f"    ✅ Matched: {chap_num}. {title} -> Book Page {logical_page} => PDF Page {actual_pdf_page}")
                    except ValueError:
                        continue

    chapter_list = list(chapter_starts.items())
    chapter_ranges = {}
    for idx, (title, start) in enumerate(chapter_list):
        corrected_start = start + start_page_correction 
        end = chapter_list[idx + 1][1] - 1 if idx + 1 < len(chapter_list) else None
        chapter_ranges[title] = [corrected_start, end]

    config = {
        book_name: {
            "chapters": chapter_ranges
        }
    }

    output_path = "book_config_generated.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ Auto-generated config saved to {output_path}")
    return config


if __name__ == "__main__":
    config_file = "book_config.json"

    ### CHANGE: define metadata explicitly
    cls = 9
    subject = "English"
    book = "Beehive"
    pdf_path = "D:/college/sem 7/CL&NLP/Books/9/English/Beehive/beehive.pdf"

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ '{config_file}' not found. Attempting to generate config automatically...")
        config = generate_book_config_from_pdf(pdf_path, book_name=book)

        if not config[book]["chapters"]:
            print("❌ Failed to auto-generate chapter list. Please check if TOC is properly formatted.")
            exit()

    ### CHANGE: collect all chunks here
    all_chunks = []

    for chapter, (start, end) in config[book]["chapters"].items():
        print(f"\n📖 Extracting '{chapter}' ({start}-{end if end else 'end'}) ...")

        try:
            final_chunks = extract_and_chunk_chapter(
                pdf_path,
                start_page=start,
                end_page=end,
                chapter=chapter,
                chunk_size=2
            )

            if not final_chunks:
                print(f"⚠️ No story content extracted for {chapter}. Check start_page in config.")
                continue

            # add metadata for each chunk
            for ch in final_chunks:
                ch["class"] = cls
                ch["subject"] = subject
                ch["book"] = book
                all_chunks.append(ch)

            print(f"✅ {chapter}: {len(final_chunks)} chunks extracted")

        except Exception as e:
            print(f"❌ Error in chapter '{chapter}': {e}")

    ### CHANGE: save once at the end
    output_filename = f"chunks_class_{cls}_{subject}_{book}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n📚 All chapters combined: {len(all_chunks)} chunks saved to {output_filename}")
