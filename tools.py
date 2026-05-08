import hashlib
import re
import os
from dotenv import load_dotenv
# Load .env from same folder as this file — works regardless of where you run from
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib import colors

# ── Init ──────────────────────────────────────────────────────────────────────

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

GROQ_MODEL = "llama-3.1-8b-instant"

def get_groq_client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY not found. Check your .env file.")
    return Groq(api_key=key)


# ── Groq Call Helper ──────────────────────────────────────────────────────────

def call_groq(prompt: str, max_tokens: int = 1024) -> str:
    """Single helper to call Groq. All functions use this."""
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


# ── Clean Transcript ──────────────────────────────────────────────────────────

def clean_transcript(text: str) -> str:
    text = re.sub(r'\[?\d{1,2}:\d{2}(?::\d{2})?\]?', '', text)  # timestamps
    text = re.sub(r'\d+\s*seconds?', '', text, flags=re.IGNORECASE)  # "8 seconds"
    text = re.sub(r'\d+\s*minutes?,?\s*\d*\s*seconds?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\u0900-\u097F]+', '', text)  # strip Devanagari script entirely
    text = re.sub(r'\[.*?\]', '', text)  # remove [संगीत] type tags
    text = re.sub(r'\b(um|uh|like|you know|okay so|alright)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r' +', ' ', text)
    return text.strip()


# ── Chunk Transcript ──────────────────────────────────────────────────────────

def chunk_transcript(transcript: str, chunk_size: int = 500) -> list:
    words = transcript.split()
    chunks = []
    overlap = 50
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ── One Smart Groq Call — All Notes ──────────────────────────────────────────

def generate_all_notes(chunks: list, subject: str, difficulty: str = "Intermediate") -> dict:
    """
    Send all chunks to Groq in ONE call.
    Returns a dict with summary, concepts, terms, revision_bullets, exam_questions.
    """
    # Limit to 2000 words — safe for all languages
    combined_words = " ".join(chunks).split()
    combined = " ".join(combined_words[:2000])

    # Difficulty instructions
    difficulty_guide = {
        "Beginner": "Use very simple language. Avoid jargon. Explain every term as if the student has zero prior knowledge.",
        "Intermediate": "Use standard academic language. Assume basic familiarity with the subject.",
        "Advanced": "Use technical language. Include deeper insights, edge cases, and connections between concepts.",
    }.get(difficulty, "Use standard academic language.")

    prompt = f"""You are a student revision assistant. A student has given you a lecture transcript on "{subject}".
The transcript may be in any language — always generate ALL notes in English regardless.
Difficulty level: {difficulty} — {difficulty_guide}

Your job is to produce exam-ready revision notes. Respond in EXACTLY this format with these section headers:

## SUMMARY
Write a clear summary in 4-5 sentences covering the main ideas.

## KEY CONCEPTS
List 6-8 key concepts. Format each as:
**Concept Name**: explanation in 1-2 lines.

## IMPORTANT TERMS
List 6-10 important terms and definitions. Format each as:
**Term**: definition.

## REVISION BULLETS
Write 10-12 short bullet points. Each bullet = one key fact a student must remember for exams.

## EXAM QUESTIONS
Generate 5 likely exam questions a professor might ask. Mix short and long answer types.
After each question, write a model answer in 2-3 lines.
Format EXACTLY as shown below with a blank line between each Q and A:

Q: question here

A: answer here

Q: next question

A: next answer

Be concise, clear, and exam-focused. No fluff.

--- TRANSCRIPT ---
{combined}
"""

    raw = call_groq(prompt, max_tokens=2500)

    # Parse sections from response
    result = {
        "summary": "",
        "concepts": "",
        "terms": "",
        "revision_bullets": "",
        "exam_questions": "",
    }

    current_section = None
    lines = raw.split("\n")
    buffer = []

    # Flexible matching — check if any keyword appears in the line
    def detect_section(line: str):
        l = line.strip().upper()
        if "SUMMARY" in l:
            return "summary"
        if "KEY CONCEPT" in l:
            return "concepts"
        if "IMPORTANT TERM" in l:
            return "terms"
        if "REVISION BULLET" in l:
            return "revision_bullets"
        if "EXAM QUESTION" in l:
            return "exam_questions"
        return None

    for line in lines:
        detected = detect_section(line)
        if detected:
            if current_section and buffer:
                result[current_section] = "\n".join(buffer).strip()
            current_section = detected
            buffer = []
        else:
            if current_section:
                buffer.append(line)

    # Save last section
    if current_section and buffer:
        result[current_section] = "\n".join(buffer).strip()

    # Fallback — if parsing failed entirely, put everything in summary
    if not any(result.values()):
        result["summary"] = raw.strip()

    return result


# ── Memory ────────────────────────────────────────────────────────────────────

def store_memory(subject: str, content: str, metadata: dict = {}):
    collection_name = subject.lower().strip().replace(" ", "_")[:50]
    collection = chroma_client.get_or_create_collection(name=collection_name)
    embedding = embedder.encode(content).tolist()
    doc_id = hashlib.md5(content.encode()).hexdigest()
    try:
        collection.add(
            documents=[content],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[metadata]
        )
    except Exception:
        pass


def retrieve_memory(subject: str, query: str, n_results: int = 3) -> list:
    collection_name = subject.lower().strip().replace(" ", "_")[:50]
    try:
        collection = chroma_client.get_collection(name=collection_name)
        count = collection.count()
        if count == 0:
            return []
        embedding = embedder.encode(query).tolist()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, count)
        )
        return results["documents"][0] if results["documents"] else []
    except Exception:
        return []


def retrieve_memory_with_groq(subject: str, query: str) -> str:
    """Retrieve from memory and use Groq to answer the query."""
    results = retrieve_memory(subject, query)
    if not results:
        return None
    # Limit context to 1500 words to stay under token limit
    context = "\n\n".join(results)
    context_words = context.split()[:1500]
    context = " ".join(context_words)
    prompt = f"""Answer this question using only the lecture notes below.
Be clear, concise, and exam-focused. Max 5 sentences.

Question: {query}

Lecture Notes:
{context}
"""
    return call_groq(prompt, max_tokens=400)


def list_subjects() -> list:
    try:
        return [c.name.replace("_", " ").title() for c in chroma_client.list_collections()]
    except Exception:
        return []


# ── Web Search ────────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 3) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        output = []
        for r in results:
            title = r.get('title', '').strip()
            body = r.get('body', '').strip().rstrip('…').rstrip('...').strip()
            url = r.get('href', '')
            output.append(f"**{title}**\n{body}\n{url}")
        return "\n\n".join(output)
    except Exception as e:
        return f"Search failed: {str(e)}"


# ── Export PDF ────────────────────────────────────────────────────────────────

def export_to_pdf(data: dict, filename: str):
    doc = SimpleDocTemplate(
        filename, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "T", parent=styles["Title"], fontSize=20,
        textColor=colors.HexColor("#1a1a2e"), spaceAfter=10
    )
    heading_style = ParagraphStyle(
        "H", parent=styles["Heading2"], fontSize=13,
        textColor=colors.HexColor("#16213e"),
        spaceBefore=14, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "B", parent=styles["Normal"], fontSize=10,
        leading=16, textColor=colors.HexColor("#333333")
    )

    story = []
    story.append(Paragraph(f"Revision Notes — {data.get('subject', '')}", title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 12))

    sections = [
        ("Summary", data.get("summary", "")),
        ("Key Concepts", data.get("concepts", "")),
        ("Important Terms", data.get("terms", "")),
        ("Revision Bullets", data.get("revision_bullets", "")),
        ("Exam Questions", data.get("exam_questions", "")),
        ("Web Enrichment", data.get("web", "")),
    ]

    for title, content in sections:
        if content and not content.startswith("Error:"):
            story.append(Paragraph(title, heading_style))
            story.append(Spacer(1, 6))
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    # Convert **bold** markdown to ReportLab <b> tags
                    safe = (
                        line.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;")
                    )
                    # Now convert **text** → <b>text</b> after escaping
                    safe = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', safe)
                    story.append(Paragraph(safe, body_style))
            story.append(Spacer(1, 10))

    doc.build(story)
    return filename
