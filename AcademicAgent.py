#!/usr/bin/env python
# coding: utf-8

"""
CourseMap Academic Planner - V6
Cleaned & unified version.

Main pieces:
- Course catalog + degree rules loading
- Interest extraction via LLM
- Rule-based 4-year plan generator
- LangGraph conversational workflow
- CLI chat loop (only when run as a script)
"""

import json
import math
import os
import pandas as pd

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from openai import AzureOpenAI

from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langgraph.graph import StateGraph, END

from pydantic import BaseModel

from config import ENDPOINT, API_KEY, MODEL_NAME, API_VERSION


# ============================================================
# 0. GLOBAL CONFIG & UTILITIES
# ============================================================

client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
)


INDEX_DIR = "./course_index_storage_new"
COURSE_CATALOG: Dict[str, Dict[str, Any]] = {}  # filled later
COURSE_VECTOR_INDEX: Optional[VectorStoreIndex] = None


def load_text_files(folder_path):
    file_text_map = {}

    # Loop through folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".txt"):
            clean_name = filename[:-4]  # remove ".txt"
            file_path = os.path.join(folder_path, filename)

            # Read the content
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Add to map
            file_text_map[clean_name] = text

    return file_text_map


# Example use:
folder = r"degree_text/"
YALE_DEGREE = load_text_files(folder)

print("YALE_DEGREE loaded counts", len(YALE_DEGREE))

def _json_default(o):
    """Helper for json.dumps to handle dataclasses, etc."""
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


# ============================================================
# 1. COURSE CATALOG + VECTOR INDEX
# ============================================================

def parse_json_field(value):
    """Safely parse JSON in dataframe fields."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return []
    return value if value else []


def clean_value(v):
    """Convert NaN to None for JSON safety."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def clean_value_int(v):
    """Convert NaN to int or None for JSON safety."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        return int(round(float(v)))
    except Exception:
        return None


def build_clean_course_catalog(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Convert the raw dataframe into a structured catalog where:
    - duplicate course codes (across FA/SP) merge
    - allowed_terms is accumulated across rows
    - course metadata is preserved
    """

    catalog: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        listings = parse_json_field(row.get("listings"))
        professors = parse_json_field(row.get("course_professors"))
        flags = parse_json_field(row.get("course_flags"))

        if not listings:
            continue

        # Always use the first listing as the canonical code
        primary_listing = listings[0]
        primary_course_code = primary_listing.get("course_code")
        if not primary_course_code:
            continue

        subject = primary_listing.get("subject")
        number = primary_listing.get("number")

        season_code = clean_value(row.get("season_code"))
        allowed_terms = []

        if isinstance(season_code, str) and len(season_code) >= 6:
            mm = season_code[-2:]  # last two digits identify term group
            # Typical Yale codes: 01 = Fall, 03 = Spring
            if mm in ("01", "00"):  # fall groups
                allowed_terms.append("FA")
            elif mm in ("03", "02"):  # spring groups
                allowed_terms.append("SP")

        # default if parsing failed
        if not allowed_terms:
            allowed_terms = ["FA", "SP"]

        # MERGE or CREATE entry
        if primary_course_code not in catalog:
            catalog[primary_course_code] = {
                "course_code": primary_course_code,
                "subject": subject,
                "number": number,
                "school": primary_listing.get("school"),

                "title": clean_value(row.get("title")),
                "description": clean_value(row.get("description")),
                "credits": clean_value(row.get("credits")),

                "professors": professors,
                "cross_listings": listings[1:],
                "flags": flags if flags else [],

                "prerequisites": clean_value(row.get("requirements")),
                "final_exam": clean_value(row.get("final_exam")),
                "extra_info": clean_value(row.get("extra_info")),

                "ratings": {
                    "average_rating": clean_value_int(row.get("average_rating")),
                    "average_professor_rating": clean_value_int(row.get("average_professor_rating")),
                    "average_workload": clean_value_int(row.get("average_workload")),
                    "average_gut_rating": clean_value_int(row.get("average_gut_rating")),
                },

                "enrollment": {
                    "last": clean_value(row.get("last_enrollment")),
                    "consistent_professors": clean_value(row.get("last_enrollment_same_professors")),
                },

                "metadata": {
                    "course_id": clean_value(row.get("course_id")),
                    "same_course_id": clean_value(row.get("same_course_id")),
                    "same_course_and_profs_id": clean_value(row.get("same_course_and_profs_id")),
                    "season_code": season_code,
                    "section": clean_value(row.get("section")),
                    "last_offered_course_id": clean_value(row.get("last_offered_course_id")),
                    "last_updated": clean_value(row.get("last_updated")),
                },

                "allowed_terms": allowed_terms.copy(),
            }

        else:
            # existing course → extend allowed_terms without duplication
            existing = catalog[primary_course_code]
            for t in allowed_terms:
                if t not in existing["allowed_terms"]:
                    existing["allowed_terms"].append(t)

    return catalog


def build_course_vector_index(course_catalog: Dict[str, Dict[str, Any]]) -> VectorStoreIndex:
    """Build a vector index from the cleaned course catalog."""
    print("Building NEW index from cleaned course_catalog...")

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
    Settings.embed_model = embed_model

    documents: List[Document] = []

    for course_code, info in course_catalog.items():
        title = info.get("title", "") or ""
        description = info.get("description", "") or ""
        prereqs = info.get("prerequisites") or "None listed"
        credits = info.get("credits", 1.0)

        profs_text = ", ".join(p.get("name", "") for p in info.get("professors", [])) or "None"
        cross_listings_text = ", ".join(x.get("course_code", "") for x in info.get("cross_listings", [])) or "None"
        flags_text = ", ".join(info.get("flags", [])) or "None"

        ratings = info.get("ratings", {})
        ratings_text = (
            f"Avg rating: {ratings.get('average_rating')}, "
            f"Avg workload: {ratings.get('average_workload')}, "
            f"Avg professor rating: {ratings.get('average_professor_rating')}, "
            f"Gut rating: {ratings.get('average_gut_rating')}"
        )

        text = (
            f"{title}.\n"
            f"Course code: {course_code}.\n"
            f"Subject: {info.get('subject')}.\n"
            f"Credits: {credits}.\n"
            f"Professors: {profs_text}.\n"
            f"Cross-listed as: {cross_listings_text}.\n"
            f"Flags: {flags_text}.\n"
            f"Prerequisites: {prereqs}.\n"
            f"Ratings: {ratings_text}.\n"
            f"Description: {description}\n"
        )

        doc = Document(
            text=text,
            metadata={
                "course_code": course_code,
                "subject": info.get("subject"),
                "number": info.get("number"),
                "title": title,
                "prerequisites": prereqs,
                "credits": credits,
            }
        )
        documents.append(doc)

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=256, chunk_overlap=0),
            embed_model,
        ]
    )
    nodes = pipeline.run(documents=documents)

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=INDEX_DIR)

    print(f"Index built and saved to {INDEX_DIR} ({len(nodes)} nodes).")
    return index


def load_course_vector_index() -> VectorStoreIndex:
    """Load the persisted vector index."""
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
    Settings.embed_model = embed_model

    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)
    return index


def query_course_catalog_rag(query: str, top_k: int = 5) -> str:
    global COURSE_VECTOR_INDEX

    print(f"\n>>> DEBUG RAG QUERY: {query}")

    if COURSE_VECTOR_INDEX is None:
        print(">>> DEBUG: Index is None")
        return "RAG index not loaded."

    retriever = COURSE_VECTOR_INDEX.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    print(f">>> DEBUG: Retrieved {len(nodes)} nodes")

    if not nodes:
        return "No matching courses found."

    results = []

    for i, node in enumerate(nodes):
        md = node.metadata
        snippet = node.text[:180].replace("\n", " ").strip()

        # print(f"\n>>> DEBUG NODE {i}:")
        # print("Metadata:", md)
        # print("Text snippet:", snippet)

        results.append(
            f"• {md.get('course_code', 'Unknown')} — {md.get('title', 'No title')}\n"
            f"  Credits: {md.get('credits', '?')}\n"
            f"  Prerequisites: {md.get('prerequisites', 'None')}\n"
            f"  {snippet}..."
        )

    return "\n\n".join(results)


def initialize_course_data():
    """
    Load CSVs and build COURSE_CATALOG and COURSE_VECTOR_INDEX.
    """
    global COURSE_CATALOG, COURSE_VECTOR_INDEX

    print(">>> DEBUG: loading course data...")

    df_fall = pd.read_csv("courses/202503.csv")
    df_spring = pd.read_csv("courses/202601.csv")

    courses_df = pd.concat([df_fall, df_spring], ignore_index=True)

    COURSE_CATALOG = build_clean_course_catalog(courses_df)
    print(f">>> DEBUG: built COURSE_CATALOG with {len(COURSE_CATALOG)} courses, {len(df_fall)}, {len(df_spring)}, {len(courses_df)}")

    try:
        COURSE_VECTOR_INDEX = load_course_vector_index()
        print(">>> DEBUG: loaded COURSE_VECTOR_INDEX from disk")
    except FileNotFoundError:
        COURSE_VECTOR_INDEX = build_course_vector_index(COURSE_CATALOG)
        print(">>> DEBUG: built new COURSE_VECTOR_INDEX")

    # Optional: build RAG index too
    # COURSE_VECTOR_INDEX = build_course_vector_index(COURSE_CATALOG)
    # COURSE_VECTOR_INDEX = load_course_vector_index()
    # print(">>> DEBUG: built COURSE_VECTOR_INDEX")

initialize_course_data()


# ============================================================
# 2. DEGREE RULES & MAJOR INTERESTS
# ============================================================
def load_degree_rules_normalized(path: str) -> Dict[str, Dict[str, Any]]:
    rules: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            obj.setdefault("prerequisites", {"components": []})
            obj["prerequisites"].setdefault("components", [])

            obj.setdefault("major_requirements", {})
            obj["major_requirements"].setdefault("required_courses", {"all_of": []})
            obj["major_requirements"].setdefault("distributions", [])

            obj.setdefault("senior_requirement", {})
            obj.setdefault("substitutions", [])
            obj.setdefault("course_relationships", {"mutual_exclusions": [], "repeat_for_credit_limits": []})
            obj.setdefault("policies", {})
            obj.setdefault("provenance", [])

            rules[obj["program_name"]] = obj

    return rules


YALE_RULES = load_degree_rules_normalized("degree_classification_final.ndjson")

# master interest taxonomy per major
with open("master_major_interests.json", "r", encoding="utf-8") as f:
    MASTER_MAJOR_INTERESTS: Dict[str, List[str]] = json.load(f)

SUPPORTED_MAJORS = list(YALE_RULES.keys())
print("YALE RULES Loaded", len(YALE_RULES))
print("SUPPORTED_MAJORS Loaded", len(SUPPORTED_MAJORS))

def normalize_major(raw: Optional[str]) -> Optional[str]:
    """Match user-provided major to YALE_RULES key (case-insensitive)."""
    if not raw:
        return None
    s = raw.strip().lower()
    for name in SUPPORTED_MAJORS:
        if name.lower() == s:
            return name
    # If no match, return None (don't hallucinate a major)
    return None


# ============================================================
# 3. DATA MODELS FOR SCHEDULER
# ============================================================

@dataclass
class Course:
    """A single course allowed for a major."""
    id: str
    title: str = ""
    description: str = ""
    buckets: List[str] = field(default_factory=list)   # ["core", "elective", "senior"]
    is_required: bool = False
    is_senior_only: bool = False
    allowed_terms: List[str] = field(default_factory=list)  # e.g., ["FA","SP"]
    source_reason: str = ""


# @dataclass
# class PlannedCourse:
#     course_id: str
#     year: int           # 1-4
#     term: str           # "FA" or "SP"
#     reason: str         # "core", "elective_interest", etc.
@dataclass
class PlannedCourse:
    course_id: str
    year: int
    term: str
    reason: str = "unspecified"

    title: Optional[str] = None
    professor: Optional[str] = None
    description: Optional[str] = None
    subject: Optional[str] = None
    number: Optional[str] = None
    allowed_terms: Optional[List[str]] = None



@dataclass
class ScheduleConfig:
    years: int = 4
    terms_per_year: int = 2
    max_courses_per_term: int = 4
    min_courses_per_term: int = 3
    senior_year: int = 4


def _course_from_code(
    code: str,
    bucket: str,
    source_reason: str,
    course_index: Dict[str, Dict[str, Any]],
) -> Course:
    """Safe helper: only returns Course if code exists in course_index."""
    info = course_index.get(code, {})
    title = info.get("title", "") or ""
    desc = info.get("description", "") or ""
    terms = info.get("terms", []) or []
    return Course(
        id=code,
        title=title,
        description=desc,
        buckets=[bucket],
        is_required=(bucket == "core"),
        is_senior_only=False,
        allowed_terms=terms,
        source_reason=source_reason,
    )

class PlannerState(TypedDict):
    llm_thread: List[Dict[str, str]]  # memory for LLM planning steps
    messages: List[Dict[str, Any]]
    major: Optional[str]
    major_interests: List[str]
    general_interests: List[str]
    plan: Optional[Any]
    auditFindings: Optional[Any]
    audit: Optional[Dict[str, Any]]


def expand_yale_rules_to_course_list(
    major_name: str,
    yale_rules: Dict[str, Dict[str, Any]],
    course_index: Dict[str, Dict[str, Any]],
) -> List[Course]:
    """
    Deterministically expand YALE_RULES[major_name] into a list of Course objects.
    No LLM; no hallucinated courses.
    """
    rules = yale_rules.get(major_name)
    if not rules:
        raise ValueError(f"No YALE_RULES found for major: {major_name}")

    courses: Dict[str, Course] = {}

    def add_or_merge(course: Course):
        existing = courses.get(course.id)
        if existing:
            for b in course.buckets:
                if b not in existing.buckets:
                    existing.buckets.append(b)
            if course.is_required:
                existing.is_required = True
            if course.is_senior_only:
                existing.is_senior_only = True
            if course.source_reason not in existing.source_reason:
                existing.source_reason += f"; {course.source_reason}"
        else:
            courses[course.id] = course

    # 1. Core required
    required = rules.get("major_requirements", {}).get("required_courses", {})
    all_of = required.get("all_of", [])
    for item in all_of:
        if isinstance(item, str):
            if item in course_index:
                c = _course_from_code(
                    item,
                    bucket="core",
                    source_reason="required core",
                    course_index=course_index,
                )
                c.is_required = True
                add_or_merge(c)
        elif isinstance(item, dict) and "one_of" in item:
            for opt in item["one_of"]:
                if opt in course_index:
                    c = _course_from_code(
                        opt,
                        bucket="core",
                        source_reason="required core (one_of group)",
                        course_index=course_index,
                    )
                    c.is_required = True
                    add_or_merge(c)

    # 2. Distributions / electives
    distributions = rules.get("major_requirements", {}).get("distributions", [])
    for dist in distributions:
        category = dist.get("category", "distribution")
        constraints = dist.get("constraints", {})
        allowed_codes = constraints.get("allowed_courses", [])
        subject = constraints.get("subject")
        numbering = constraints.get("numbering", {})
        min_num = numbering.get("min_number", 0)
        max_num = numbering.get("max_number", 9999)

        for code in allowed_codes:
            if code in course_index:
                c = _course_from_code(
                    code,
                    bucket="elective",
                    source_reason=f"distribution: {category}",
                    course_index=course_index,
                )
                add_or_merge(c)

        if subject:
            for code in course_index.keys():
                try:
                    subj, num_str = code.split()
                    num = int(num_str)
                except Exception:
                    continue
                if subj == subject and min_num <= num <= max_num:
                    c = _course_from_code(
                        code,
                        bucket="elective",
                        source_reason=f"distribution ({category} via subject/number range)",
                        course_index=course_index,
                    )
                    add_or_merge(c)

    # 3. Senior requirement
    sr = rules.get("senior_requirement", {})
    options = sr.get("options", {})
    for opt_name, opt in options.items():
        for code in opt.get("courses", []):
            if code in course_index:
                c = _course_from_code(
                    code,
                    bucket="senior",
                    source_reason=f"senior requirement ({opt_name})",
                    course_index=course_index,
                )
                c.is_senior_only = True
                add_or_merge(c)

    return list(courses.values())


def compute_interest_score_for_course(
    course: Course,
    major_interests: List[str],
    general_interests: List[str],
) -> float:
    """Simple lexical match scoring."""
    text = (course.title + " " + course.description).lower()
    score = 0.0

    for mi in major_interests:
        mi = mi.lower().strip()
        if mi and mi in text:
            score += 2.0

    for gi in general_interests:
        gi = gi.lower().strip()
        if gi and gi in text:
            score += 1.0

    if score > 0 and "core" in course.buckets:
        score += 0.5

    return score

def rag_search_interests(major_interests, general_interests, top_k=10):
    """
    Semantic search for interest-based courses using the vector index.
    Returns a list of RAG nodes.
    """
    results = []
    all_interests = major_interests + general_interests

    for interest in all_interests:
        print(f">>> DEBUG RAG interest search: {interest}")
        try:
            rag_hits = COURSE_VECTOR_INDEX.as_retriever(
                similarity_top_k=top_k
            ).retrieve(interest)

            print(f">>> DEBUG: {len(rag_hits)} courses found for interest '{interest}'")
            results.extend(rag_hits)

        except Exception as e:
            print(">>> RAG ERROR:", e)

    return results

def convert_rag_nodes_to_structured_courses(rag_nodes):
    """
    Convert RAG search results into the structured format required by COURSE_CATALOG.
    Robust to messy metadata (non-digit course numbers, missing fields).
    """
    structured = {}

    for node in rag_nodes:
        md = node.metadata

        code = md.get("course_code")
        if not code:
            continue

        # -------------------------
        # SAFE subject & number parse
        # -------------------------
        parts = code.split()
        if len(parts) >= 2:
            subject = parts[0]
            raw_number = parts[1]
        else:
            subject = md.get("subject", "UNKNOWN")
            raw_number = md.get("number", "0")

        # Extract only digits from number e.g. "7716O" → "7716"
        digits = "".join([c for c in raw_number if c.isdigit()])
        number = int(digits) if digits else 0

        # -------------------------
        # SAFE structured course dict
        # -------------------------
        structured[code] = {
            "code": code,
            "title": md.get("title", ""),
            "description": node.text or "",
            "subject": subject,
            "number": number,
            "credits": md.get("credits", 1),
            "prerequisites": md.get("prerequisites", []),
            "allowed_terms": md.get("allowed_terms", ["FA", "SP"]),
            "buckets": md.get("buckets", ["elective"]),
            "areas": md.get("areas", []),
        }

    print(f">>> DEBUG: Structured {len(structured)} RAG-obtained courses")
    return structured


def merge_rag_courses_into_catalog(structured_rag_courses):
    """
    Merge semantic RAG course hits into the COURSE_CATALOG
    so that the planner can use them as electives.
    """
    global COURSE_CATALOG

    added = 0
    for code, info in structured_rag_courses.items():
        if code not in COURSE_CATALOG:
            COURSE_CATALOG[code] = info
            added += 1

    print(f">>> DEBUG: Added {added} new courses from RAG into COURSE_CATALOG")

def rank_courses_by_interest(
    allowed_courses: List[Course],
    major_interests: List[str],
    general_interests: List[str],
) -> List[Tuple[Course, float]]:
    scored: List[Tuple[Course, float]] = []
    for c in allowed_courses:
        s = compute_interest_score_for_course(c, major_interests, general_interests)
        scored.append((c, s))

    scored.sort(
        key=lambda cs: (
            cs[1],
            "core" in cs[0].buckets,
            cs[0].title.lower(),
        ),
        reverse=True,
    )
    return scored


def build_course_pool_for_llm(
    state: PlannerState,
    course_index: Dict[str, Dict[str, Any]],
    major_name: str,
    major_interests: List[str],
    general_interests: List[str],
    max_courses: int = 250,
) -> List[Dict[str, Any]]:
    """
    Build a filtered pool of real Yale courses that the LLM can choose from.
    We don't send the entire catalog – just a representative subset:
      - courses in the major department (e.g. CPSC for CS)
      - courses related to general interests (writing, photography, etc.)
      - optional RAG hits
    Each course dict is a lightweight view: code, title, description, allowed_terms, etc.
    """

    # crude mapping from major name to subject prefix
    MAJOR_SUBJECT_MAP = {
    'Anthropology': 'ANTH',
    'African Studies': 'AFST',
    'Aerospace Studies': 'USAF',
    'African American Studies': 'AFAM',
    'American Studies': 'AMST',
    'Applied Mathematics': 'AMTH',
    'Applied Physics': 'APHY',
    'Archaeological Studies': 'ARCG',
    'Architecture': 'ARCH',
    'Art': 'ART',
    'Astronomy': 'ASTR',
    'Biology': 'BIOL',
    'British Studies': 'BRST',
    'Astrophysics': 'ASTR',
    'Biomedical Engineering': 'BENG',
    'Chemical Engineering': 'CENG',
    'Engineering Sciences (Chemical)': 'ENAS',
    'Chemistry': 'CHEM',
    'Child Study': 'CHLD',
    'Classics': 'CLSS',
    'Climate Science and Solutions Certificate': 'EPS',
    'Classical Civilization': 'CLCV',
    'Ancient and Modern Greek': 'GREK',
    'College Seminars Program': None,
    'Comparative Literature': 'CPLT',
    'Cognitive Science': 'CGSC',
    'Computer Science and Economics': 'CSEC',
    'Computer Science': 'CPSC',
    'Computer Science and Mathematics': 'CPSC',
    'Computer Science and Psychology': 'CPSC',
    'Certificate in Programming': 'CPSC',
    'DeVane Lecture Course': 'DEVN',
    'Computing and Linguistics': 'CSLI',
    'Directed Studies': 'DRST',
    'Computing and the Arts': 'CPAR',
    'Earth and Planetary Sciences': 'EPS',
    'East Asian Languages and Literatures': 'EALL',
    'East Asian Studies': 'EAST',
    'Economics and Mathematics': 'ECON',
    'Ecology and Evolutionary Biology': 'EEB',
    'Education Studies Scholar Intensive Certificate': 'EDST',
    'Economics': 'ECON',
    'Electrical Engineering and Computer Science': None,
    'Electrical Engineering (ABET)': 'EENG',
    'Engineering Sciences (Electrical), B.S.': 'ENAS',
    'Certificate in Education Studies': None,
    'Engineering Sciences (Electrical), B.A.': 'ENAS',
    'Energy Studies Certificate': 'ENRG',
    'English Language and Literature': 'ENGL',
    'Environmental Studies': 'EVST',
    'Engineering Sciences (Environmental)': 'ENVE',
    'Environmental Engineering': 'ENVE',
    'First-Year Seminar Program': None,
    'Ethnicity, Race, and Migration': 'ER&M',
    'Ethics, Politics, and Economics': 'EP&E',
    'Food, Agriculture, and Climate Change Certificate': 'EVST',
    'French': 'FREN',
    'Ethnography Certificate': None,
    'Global Health Studies Certificate': 'HLTH',
    'Film and Media Studies': 'FILM',
    'Hellenic Studies': None,
    'German Studies': 'GMAN',
    'Global Affairs': 'GLBL',
    'Intensive Certificate in Human Rights': 'HMRT',
    'History of Art': 'HSAR',
    'History of Science, Medicine, and Public Health': 'HSHM',
    'Certificate in Human Rights': 'HMRT',
    'Humanities': 'HUMS',
    'History': 'HIST',
    'Islamic Studies Certificate': None,
    'Italian Studies': 'ITAL',
    'Medieval Studies Certificate': None,
    'Latin American Studies': 'LAST',
    'Linguistics': 'LING',
    'Mathematics and Physics': None,
    'Jewish Studies': 'JDST',
    'Mathematics': 'MATH',
    'Mathematics and Philosophy': None,
    'Engineering Sciences (Mechanical)': 'ENAS',
    'Mechanical Engineering': 'MENG',
    'Modern Middle East Studies': 'MMES',
    'Naval Science': 'NAVY',
    'Near Eastern Languages and Civilizations': 'NELC',
    'Molecular, Cellular, and Developmental Biology': 'MCDB',
    'Music': 'MUSI',
    'Persian and Iranian Studies Certificate': None,
    'Neuroscience': 'NSCI',
    'Molecular Biophysics and Biochemistry': 'MB&B',
    'Philosophy': 'PHIL',
    'Physics and Philosophy': None,
    'Physics and Geosciences': None,
    'Portuguese': 'PORT',
    'Certificate of Advanced Language Study in Portuguese': None,
    'Political Science': 'PLSC',
    'Quantum Science and Engineering Certificate': None,
    'Physics': 'PHYS',
    'B.A.–B.S./M.P.P. Program in Global Affairs': None,
    'Religious Studies': 'RLST',
    'Russian': 'RUSS',
    'Science': 'SCIE',
    'Psychology': 'PSYC',
    'B.A.–B.S./M.E.M. or M.E.Sc. Five-Year Joint Degree Program': None,
    'B.A.–B.S./M.P.H. Five-Year Program in Public Health': None,
    'South Asian Studies': 'SAST',
    'Sociology': 'SOCY',
    'Spanish': 'SPAN',
    'Certificate of Advanced Language Study in Indonesian': 'INDN',
    'Certificate of Advanced Language Study in Vietnamese': 'VIET',
    'Translation Studies Certificate': 'TRAN',
    'Theater, Dance, and Performance Studies': 'TDPS',
    'Special Divisional Major': 'SPEC',
    "Women's, Gender, and Sexuality Studies": 'WGSS',
    'Urban Studies': 'URBN',
    'Statistics and Data Science': 'S&DS',
    'Data Science Certificate': 'S&DS'
    }


    subject_prefixes = [MAJOR_SUBJECT_MAP.get(major_name)]

    pool: Dict[str, Dict[str, Any]] = {}

    # 1) all courses in the major subject(s)
    for code, info in course_index.items():
        subj = info.get("subject") or code.split()[0]
        if subj in subject_prefixes:
            pool[code] = info

    # 2) simple interest-based filter on catalog text
    interest_keywords = [kw.lower() for kw in (major_interests + general_interests)]
    for code, info in course_index.items():
        if len(pool) >= max_courses:
            break
        title = (info.get("title") or "").lower()
        desc = (info.get("description") or "").lower()
        if any(kw in title or kw in desc for kw in interest_keywords):
            pool.setdefault(code, info)

    # 3) optional RAG enrichment (if you want to fold in semantic matches)
    try:
        for interest in (major_interests + general_interests):
            if len(pool) >= max_courses:
                break
            rag_text = query_course_catalog_rag(interest, top_k=8)
            # if query_course_catalog_rag returns formatted text, skip;
            # if you instead have a raw node-based retriever, you can adapt that here.
            # For now we assume course_index already covers most of those courses.
    except Exception as e:
        print(">>> WARNING: RAG enrichment failed in build_course_pool_for_llm:", e)

    # 4) trim to max_courses
    pool_list: List[Dict[str, Any]] = []
    for code, info in pool.items():
        if len(pool_list) >= max_courses:
            break

        allowed_terms = info.get("allowed_terms", ["FA", "SP"])
        pool_list.append(
            {
                "course_code": code,
                "title": info.get("title", ""),
                "description": info.get("description", ""),
                "subject": info.get("subject", code.split()[0]),
                "number": info.get("number", 0),
                "allowed_terms": allowed_terms,
                "prerequisites": info.get("prerequisites", ""),
            }
        )

    print(f">>> LLM COURSE POOL SIZE: {len(pool_list)}")
    return pool_list



def llm_call(
    state: PlannerState,
    system_msg: str,
    user_msg: str,
    user_input: str
):
    """
    LLM call with clean memory strategy:
    - user_input = raw natural language typed by the user
    - user_msg   = technical JSON / prompt for this specific call
    - system_msg = system prompt for this specific call

    Only user_input and natural-language assistant replies are stored
    in history. All technical JSON/system data is ephemeral.
    """

    # ---- 1. Load short conversation history (natural language only) ----
    conv_thread = state["llm_thread"]

    # ---- 2. Build messages for the actual LLM call ----
    messages = []

    # System prompt is ALWAYS included per call
    messages.append({"role": "system", "content": system_msg})

    # Add *only* the natural-language history
    messages.extend(conv_thread)

    # Add the technical user payload (JSON)
    messages.append({"role": "user", "content": user_msg})

    # ---- 3. Make the LLM call ----
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )

    ai_text = response.choices[0].message.content

    # ---- 4. Store only the NATURAL-LANGUAGE parts ----

    # Store what the user ACTUALLY typed
    if user_input and user_input.strip():
        conv_thread.append({"role": "user", "content": user_input})

    # Only store assistant message if it's not JSON
    if not ai_text.strip().startswith("{"):
        conv_thread.append({"role": "assistant", "content": ai_text})

    return ai_text


def llm_build_4_year_schedule(
    state: PlannerState,
    major_name: str,
    major_interests: List[str],
    general_interests: List[str],
    course_pool: List[Dict[str, Any]],
    yale_rules_for_major: Dict[str, Any],
    config: ScheduleConfig,
) -> Tuple[List[PlannedCourse], List[str]]:
    """
    Ask the LLM to lay out the previously selected courses term-by-term over 4 years.
    We pass back the same course_pool (with allowed_terms, etc.), and the selected
    course code sets per category.
    Returns a list of PlannedCourse objects.
    """

    system_msg = (
        "You are an expert Yale academic scheduler.\n"
        "Your job: build a COMPLETE 4-year term-by-term schedule.\n\n"

        "You are given:\n"
        "- A pool of real Yale courses (with allowed_terms and descriptions).\n"
        "- A JSON description of degree requirements for a major.\n"
        "and sometimes text describing the rule for a degree as well\n"
        " The student's interests in the major and also out side the major, some of the general interests "
        "- A schedule config with 4 years × 2 terms (FA/SP) and max courses per term.\n\n"

        "Your responsibilities:\n"
        "1. Use ALL required, core, senior, and interest-aligned courses.\n"
        "2. Fill EVERY one of the 8 terms with ~4 courses (approx 32 total).\n"
        "3. If the selected course lists are not enough to fill 8 terms, "
        "   choose additional appropriate electives from the course_pool "
        "   that satisfy Yale's distribution expectations: humanities/arts, social science, "
        "   science, writing, quantitative reasoning.\n"
        "4. Respect each course's allowed_terms (FA/SP only).\n"
        "5. Respect prerequisites logically (intro first, senior last).\n"
        "6. Spread workload and difficulty sensibly across terms.\n"
        "7. Create a short audit report per the degree rules, at the starting of each sentense it should say MET or NOT MET indicating if these conditions have been met or not\n"
        "   a) Are all requirements for major included in plan?\n"
        "   b) Are all course prereqs for the courses included in plan, its like recursive checking to see if the course has a prereq which is also included in plan?\n"
        "   c) Are all distributional requirements met?\n"
        "8. No invented courses — only course_code from the course_pool.\n"

        "Return STRICT JSON ONLY:\n"
        "{'courses': [{\"course_code\": \"CPSC 2010\", \"year\": 1, \"term\": \"FA\", \"reason\": \"core\"}], 'audit':['finding 1',' finding 2']}"
    )

    user_msg = {
        "schedule_config": {
            "years": config.years,
            "terms_per_year": config.terms_per_year,
            "max_courses_per_term": config.max_courses_per_term,
            "senior_year": config.senior_year,
        },
        "course_pool": course_pool,
        "major_name": major_name,
        "major_interests": major_interests,
        "general_interests": general_interests,
        "yale_rules_for_major": yale_rules_for_major,
        "yale_text_rules_for_major": YALE_DEGREE[major_name],
        "term_labels": [
            {"year": y, "term": term}
            for y in range(1, config.years + 1)
            for term in ("FA", "SP")
        ],
    }


    raw = llm_call(state, system_msg, json.dumps(user_msg), "")
    print(">>> DEBUG LLM schedule raw:", raw[:400])
    catalog_map = {c["course_code"]: c for c in course_pool}
    try:
        schedule_json = json.loads(raw)
    except Exception as e:
        print(">>> ERROR parsing LLM schedule JSON:", e)
        return [], []

    planned: List[PlannedCourse] = []
    course_parts = schedule_json['courses']
    audit_parts = schedule_json['audit']
    print("Returned Audit:", json.dumps(audit_parts))
    for item in course_parts:
        try:
            pc = PlannedCourse(
                course_id=item["course_code"],
                year=int(item["year"]),
                term=item["term"],
                reason=item.get("reason", "unspecified"),
            )

            # ---- NEW: ENRICH USING COURSE CATALOG ----
            if pc.course_id in catalog_map:
                meta = catalog_map[pc.course_id]
                pc.title = meta.get("title")
                pc.professor = meta.get("professor")  # If your catalog contains this
                pc.description = meta.get("description")
                pc.subject = meta.get("subject")
                pc.number = meta.get("number")
                pc.allowed_terms = meta.get("allowed_terms")

            planned.append(pc)

        except Exception as e:
            print(">>> ERROR converting schedule item to PlannedCourse:", e, "item:", item)

    return planned, audit_parts



def generate_4_year_plan(
    state: PlannerState,
    major_name: str,
    major_interests: List[str],
    general_interests: List[str],
    yale_rules: Dict[str, Dict[str, Any]],
    course_index: Dict[str, Dict[str, Any]],
    config: Optional[ScheduleConfig] = None,
) -> Tuple[List[PlannedCourse], List[str]]:
    """
    NEW LLM-ASSISTED PLANNER:
      1) Build a course pool from COURSE_CATALOG (and optional RAG).
      2) Ask LLM which courses from that pool should fulfill the degree requirements
         and reflect the student's interests.
      3) Ask LLM to lay those courses out over 4 years, respecting allowed_terms, etc.
      4) Return a list of PlannedCourse.
    """

    if config is None:
        config = ScheduleConfig()

    print("\n================== LLM PLANNER ==================")
    print("Major:", major_name)
    print("Major interests:", major_interests)
    print("General interests:", general_interests)
    print("Catalog size:", len(course_index))

    rules_for_major = yale_rules.get(major_name, {})
    if not rules_for_major:
        print(">>> WARNING: No YALE_RULES entry for major:", major_name)

    # # 1) Build the course pool
    course_pool = build_course_pool_for_llm(
        state=state,
        course_index=course_index,
        major_name=major_name,
        major_interests=major_interests,
        general_interests=general_interests,
        max_courses=500,
    )

    if not course_pool:
        print(">>> ERROR: Course pool is empty; cannot build plan.")
        return [], []

    print(">>> DEBUG: course pool is loaded", len(course_pool))


    # 2) LLM builds the actual 4-year schedule from those courses
    planned, audit_parts = llm_build_4_year_schedule(
        state=state,
        major_name=major_name,
        major_interests=major_interests,
        general_interests=general_interests,
        course_pool=course_pool,
        yale_rules_for_major=rules_for_major,
        config=config,
    )

    print(">>> LLM PLANNER: produced", len(planned), "PlannedCourse items")
    return planned, audit_parts


def test_generate_plan(
    major="Computer Science",
    major_interests=["artificial intelligence", "software engineering"],
    general_interests=["creative writing", "photography"],
    top_k=10
):
    mock_state = {
    "llm_thread": [
        {"role": "system", "content": "You are an expert Yale academic planning LLM."}
    ],
    "messages": [],
    "major": major,
    "major_interests": major_interests,
    "general_interests": general_interests,
    "plan": None,
    "audit": None,
    }
    print("===============================================")
    print("🔬 TEST: Running generate_4_year_plan manually")
    print("===============================================\n")

    print("Major:", major)
    print("Major Interests:", major_interests)
    print("General Interests:", general_interests)
    print("COURSE_CATALOG size:", len(COURSE_CATALOG))
    print("RAG Index Loaded:", COURSE_VECTOR_INDEX is not None)
    print()

    # ---------------------------------------------------
    # Run the planner manually (bypassing LangGraph)
    # ---------------------------------------------------
    plan, audit_parts = generate_4_year_plan(
        state=mock_state,
        major_name=major,
        major_interests=major_interests,
        general_interests=general_interests,
        yale_rules=YALE_RULES,
        course_index=COURSE_CATALOG,
        config=ScheduleConfig()
    )

    # ---------------------------------------------------
    # Display results
    # ---------------------------------------------------
    print("\n===============================================")
    print("PLAN RESULT")
    print("===============================================\n")

    if not plan:
        print("No courses were scheduled!")
        print("\nLikely reasons:")
        print("- COURSE_CATALOG entries missing allowed_terms")
        print("- RAG courses missing buckets/terms")
        print("- major_interests/general_interests didn’t match course descriptions")
        print("- rule expansion returned zero courses")
        return

    print(f"Number of planned courses: {len(plan)}\n")

    # pretty print first 20 entries
    for pc in plan:
        course = COURSE_CATALOG.get(pc.course_id, {})
        print(f"{pc.year} {pc.term}   {pc.course_id}   ({pc.reason})   {course.get('title','?')}")
    for ads in audit_parts:
        print(f"Audit Finding: {ads}")

    print("\n===============================================")
    print("END OF TEST")
    print("===============================================")


# ============================================================
# 4. GOAL EXTRACTION VIA LLM
# ============================================================

class GoalExtraction(BaseModel):
    major: Optional[str] = None
    major_interests: List[str] = []
    general_interests: List[str] = []


class Pass1Schema(BaseModel):
    major: Optional[str] = None
    raw_interests: List[str] = []


PASS1_PROMPT = """
Extract the following from the student's message.

Return this JSON ONLY:

{{
  "major": string | null,
  "raw_interests": string[]
}}

Definitions:
- major: a declared or strongly implied Yale major from this list:
  {supported_majors}
- raw_interests: the user's stated interests EXCLUDING the major name itself.
- Do NOT classify or filter. Just extract.

User message:
{user_message}
"""

PASS2_PROMPT = """
You are an academic classifier for Yale student interests.

Your job is to map the student’s stated items into:
1. major_interests  (ONLY official subfields of the major)
2. general_interests (ONLY academic subjects outside the major)

MAJOR: {major}

OFFICIAL SUBFIELDS (canonical names):
{master_interests}

USER PROVIDED RAW ITEMS:
{raw_interests}

======================================================
INTERPRETATION RULES (EXTREMELY IMPORTANT)
======================================================

1. **Fuzzy Matching Allowed**
   You MUST allow semantic/fuzzy matching between a raw item and a subfield.
   This includes:
   - synonyms (“AI” → “Artificial Intelligence”),
   - abbreviations,
   - variants (“cognitive neuroscience” → “Cognitive Neuroscience”),
   - related descriptions (“programming and simulation” → “Computational Modeling”)

2. **Canonicalization**
   Always map to the closest **canonical** subfield name from the official list.
   NEVER create new subfields.

3. **General Interests = ONLY academic disciplines OUTSIDE the major**
   Examples of valid: biology, art history, philosophy, political science, mathematics, neuroscience (if major is not related), literature.

4. **Hard EXCLUSIONS**
   EXCLUDE completely if the item is:
   - a career goal (industry, graduate school, job)
   - software/tools (Python, MATLAB, Blender, TensorFlow)
   - a hobby (sports, music, gym, clubs)
   - planning/requests (“I need a roadmap”)
   - experience (“I did research in…”)
   - skills (“coding”, “leadership”, “public speaking” unless an academic field)
   - personal background
   - vague phrases (“learning”, “making things”)

======================================================
STRICT ANTI-LEAKAGE RULES
======================================================

Before returning output:
- NOTHING that matches (even loosely) a major subfield may appear in general_interests.
- NOTHING that does not represent a real academic discipline may appear anywhere.
- If an item belongs to neither category → EXCLUDE it.

======================================================
RETURN JSON ONLY
======================================================

Use this exact structure:

{{
  "major": string,
  "major_interests": [
      // canonical subfield names
  ],
  "general_interests": [
      // academic subjects outside the major
  ]
}}

"""



def llm_extract_goals(state: PlannerState, user_message: str) -> GoalExtraction:
    # PASS 1
    prompt1 = PASS1_PROMPT.format(
        supported_majors=", ".join(SUPPORTED_MAJORS),
        user_message=user_message,
    )

    text1 = llm_call(state, "Extract academic intentions as JSON.", prompt1, user_message)    

    try:
        data1 = json.loads(text1)
    except Exception:
        data1 = {"major": None, "raw_interests": []}

    pass1 = Pass1Schema(**data1)
    major = normalize_major(pass1.major)

    # No major → everything is general
    if major is None:
        return GoalExtraction(
            major=None,
            major_interests=[],
            general_interests=[i.lower() for i in pass1.raw_interests],
        )

    # PASS 2
    master = MASTER_MAJOR_INTERESTS.get(major, [])
    prompt2 = PASS2_PROMPT.format(
        major=major,
        master_interests=json.dumps(master, indent=2),
        raw_interests=json.dumps(pass1.raw_interests, indent=2),
    )

    text2 = llm_call(state, "Classify interests according to the major’s subfields.", prompt2, user_message)    
    try:
        data2 = json.loads(text2)
    except Exception:
        data2 = {"major": major, "major_interests": [], "general_interests": []}

    final = GoalExtraction(**data2)

    return GoalExtraction(
        major=major,
        major_interests=[i.lower() for i in final.major_interests],
        general_interests=[i.lower() for i in final.general_interests],
    )


# ============================================================
# 5. STATE, GRAPH NODES & WORKFLOW
# ============================================================




def ingest_user_message(state: PlannerState) -> PlannerState:
    """Currently a no-op; user message already appended in chat_with_agent."""
    return state


def extract_goals(state: PlannerState) -> PlannerState:
    messages = state["messages"]

    # find last user message
    last_msg = None
    for m in reversed(messages):
        if m.get("type") == "user":
            last_msg = m.get("content")
            break

    if not last_msg:
        return state

    result = llm_extract_goals(state, last_msg)

    if result.major:
        state["major"] = result.major

    state["major_interests"] = list({
        *state.get("major_interests", []),
        *[i.lower() for i in (result.major_interests or [])],
    })

    state["general_interests"] = list({
        *state.get("general_interests", []),
        *[i.lower() for i in (result.general_interests or [])],
    })

    return state

RAG_ENABLED_QUERIES = [
    "find courses",
    "find me courses",
    "search for courses",
    "recommend courses",
    "electives",
    "ai electives",
    "ml electives",
    "writing courses",
    "photography courses",
    "computer science courses",
    "show me courses",
]

def agent_interact(state: PlannerState) -> PlannerState:
    messages = state["messages"]
    major = state.get("major")
    major_interests = state.get("major_interests", [])
    general_interests = state.get("general_interests", [])
    plan = state.get("plan")

    # last user message
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("type") == "user":
            last_user_msg = m.get("content", "")
            break

    lowered = last_user_msg.lower()
    if any(keyword in lowered for keyword in RAG_ENABLED_QUERIES):
        print(">>> DEBUG agent_interact: RAG triggered")

        rag_response = query_course_catalog_rag(last_user_msg)

        state["messages"].append({
            "type": "ai",
            "content": (
                "Here are some courses I found:\n\n" +
                rag_response +
                "\n\nYou can ask for more details or a 4-year plan."
            )
        })
        return state


    # decision logic for system prompt
    if plan is not None:
        user_goal_instruction = (
            "You already have a structured 4-year academic plan in context['plan'].\n"
            "Your job:\n"
            "- Summarize the plan term-by-term.\n"
            "- Highlight how it reflects their interests.\n"
            "- If there are any Audit Findings then mention them in the output.\n"
            "- Invite follow-up questions.\n"
            "IMPORTANT: Do NOT generate a new plan.\n"
        )
    elif not major:
        user_goal_instruction = (
            "The student has NOT specified a major.\n"
            "Ask which Yale major they want a 4-year plan for.\n"
            "Do NOT guess. Do NOT generate a plan yet.\n"
        )
    elif major and not major_interests:
        user_goal_instruction = (
            f"The student chose {major} but has not given any interests within the major.\n"
            "Ask what areas inside the major they care about (topics, subfields, etc.).\n"
        )
    elif major_interests and not general_interests:
        pretty = ", ".join(major_interests)
        user_goal_instruction = (
            f"The student is majoring in {major} with interests: {pretty}.\n"
            "Ask about interests outside the major (electives, humanities, arts, clubs, etc.).\n"
        )
    else:
        pretty_major = ", ".join(major_interests)
        pretty_general = ", ".join(general_interests)
        user_goal_instruction = (
            f"Student: major={major}, major_interests={pretty_major}, general_interests={pretty_general}.\n"
            "Continue the conversation naturally.\n"
            "Offer help building or refining a 4-year plan if they ask.\n"
            "Do NOT repeatedly ask for info we already have.\n"
        )

    system_prompt = (
        "You are Yalemate, a Yale academic planning assistant.\n"
        "Be concise, friendly, and focused on Yale academics.\n\n"
        + user_goal_instruction
    )

    context = {
        "major": major,
        "major_interests": major_interests,
        "general_interests": general_interests,
        "plan": plan,
        "auditFindings": state.get("auditFindings"),
    }

    llm_input = (
        f"CONTEXT:\n{json.dumps(context, indent=2, default=_json_default)}\n\n"
        f"USER MESSAGE:\n{last_user_msg}"
    )

    ai_text = llm_call(
        state,
        system_prompt,
        llm_input, ""
    )

    state["messages"].append({"type": "ai", "content": ai_text})
    return state


def user_requests_plan(msg: str) -> bool:
    m = msg.lower()
    triggers = [
        "4 year plan", "four year plan",
        "generate plan", "generate my plan",
        "build my plan", "make my plan",
        "course plan", "academic plan",
        "schedule my courses", "make a schedule",
    ]
    return any(t in m for t in triggers)


def ready_for_planning(state: PlannerState) -> bool:
    return (
        bool(state.get("major")) and
        len(state.get("major_interests", [])) > 0 and
        len(state.get("general_interests", [])) > 0
    )


def generate_plan_node(state: PlannerState) -> PlannerState:
    print(">>> DEBUG: entering plan generator")

    # --- FIX: call the new generate_4_year_plan that takes state ---
    plan, audit_finds = generate_4_year_plan(
        state,
        major_name=state.get("major"),
        major_interests=state.get("major_interests", []),
        general_interests=state.get("general_interests", []),
        yale_rules=YALE_RULES,
        course_index=COURSE_CATALOG,
        config=ScheduleConfig()
    )

    plan_dict = []
    for pc in plan:
        plan_dict.append({
            "course_id": pc.course_id,
            "year": pc.year,
            "term": pc.term,
            "reason": pc.reason,
            
            # --- NEW METADATA FIELDS ---
            "title": pc.title,
            "professor": pc.professor,
            "description": pc.description,
            "subject": pc.subject,
            "number": pc.number,
            "allowed_terms": pc.allowed_terms,
        })

    if (audit_finds and len(audit_finds) > 0):
        state['auditFindings'] = audit_finds
    # Save to state
    state["plan"] = plan_dict

    return state




def to_interact_or_plan(state: PlannerState) -> str:
    messages = state["messages"]

    if messages and messages[-1].get("type") == "ai":
        return "interact"

    last_user = None
    for m in reversed(messages):
        if m.get("type") == "user":
            last_user = m.get("content")
            break

    # No user message? Just continue normally.
    if not last_user:
        return "interact"

    if state.get("_last_routed_user_msg") == last_user:
        return "interact"

    # Update last routed user message
    state["_last_routed_user_msg"] = last_user

    # Main routing logic
    if user_requests_plan(last_user):
        return "generate_plan"

    if ready_for_planning(state):
        return "generate_plan"

    return "interact"

def main_node(state: PlannerState) -> PlannerState:
    """
    Single-step node that:
    - Extracts goals from the latest user message
    - Updates state (major, interests)
    - Either chats OR generates a plan
    - Then returns updated state (no recursion, no cycles)
    """

    messages = state["messages"]

    # 1) Find last user message
    last_user = None
    for m in reversed(messages):
        if m.get("type") == "user":
            last_user = m.get("content")
            break

    if not last_user:
        return state

    # 2) Extract goals
    state = extract_goals(state)

    # 3) Decide plan vs chat
    if user_requests_plan(last_user) or ready_for_planning(state):
        print(">>> DEBUG: main_node → generate_plan_node")
        state = generate_plan_node(state)

        # After generating a plan, the agent must respond with a summary.
        print(">>> DEBUG: main_node → agent_interact (after plan)")
        state = agent_interact(state)

    else:
        print(">>> DEBUG: main_node → agent_interact")
        state = agent_interact(state)

    return state


# Build the LangGraph workflow
# -------------------------------
# WORKFLOW GRAPH (FINAL FIX)
# -------------------------------

# -------------------------------
# WORKFLOW GRAPH (NO CYCLES)
# -------------------------------

workflow = StateGraph(PlannerState)

# Only one node: main_node
workflow.add_node("main", main_node)

# Entry point is main
workflow.set_entry_point("main")

# main → END (no loops)
workflow.add_edge("main", END)

agent = workflow.compile()



# ============================================================
# 6. CHAT WRAPPER + CLI
# ============================================================

# Global state for interactive use
state: PlannerState = {
    "messages": [],
    "llm_thread": [
        {"role": "system", "content": "You are an expert Yale academic planning LLM."}
    ],
    "major": None,
    "major_interests": [],
    "general_interests": [],
    "plan": None,
    "audit": None,
}


# def chat_with_agent(user_input: str) -> str:
#     global state
#     return chat_with_agent(state, user_input)

def chat_with_agent(state:PlannerState,  user_input: str) -> Tuple[str, Any]:
    # global state

    # DEBUG — BEFORE INGEST
    print("\n================ DEBUG: BEFORE INGEST ================")
    print(json.dumps({
        "major": state.get("major"),
        "major_interests": state.get("major_interests"),
        "general_interests": state.get("general_interests"),
        "plan_exists": state.get("plan") is not None,
        "messages_count": len(state["messages"]),
    }, indent=2))

    # Append the user message before calling graph
    state["messages"].append({"type": "user", "content": user_input})

    # DEBUG — AFTER INGEST
    print("\n================ DEBUG: AFTER INGEST =================")
    print(json.dumps({
        "last_user_message": user_input,
        "messages_count": len(state["messages"]),
    }, indent=2))

    # Invoke LangGraph agent
    state = agent.invoke(state, config={"recursion_limit": 3})

    # DEBUG — AFTER AGENT
    print("\n================ DEBUG: AFTER AGENT =================")
    print(json.dumps({
        "major": state.get("major"),
        "major_interests": state.get("major_interests"),
        "general_interests": state.get("general_interests"),
        "plan_exists": state.get("plan") is not None,
        "messages_count": len(state["messages"]),
        "auditFindings": state.get("auditFindings") is not None,
    }, indent=2))

    ai_messages = [m for m in state["messages"] if m.get("type") == "ai"]
    if not ai_messages:
        return "⚠️ No AI response generated."

    response = ai_messages[-1]["content"]

    print("\n================ DEBUG: AI RESPONSE ==================")
    print(response)
    print("======================================================")

    return response, state


def interactive_chat():
    print("\n🎓  Welcome to Yalemate Academic Planner")
    print("Type your questions or planning preferences below.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("👤 You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        response = chat_with_agent(user_input)
        print(f"\n🤖 Yalemate:\n{response}\n")


# ============================================================
# 7. BOOTSTRAP COURSE_CATALOG / INDEX (OPTIONAL)
# ============================================================

def initialize_course_data():
    """
    Hook to load your CSVs and build COURSE_CATALOG and COURSE_VECTOR_INDEX.

    Call this once at startup, before interactive_chat().
    """
    global COURSE_CATALOG, COURSE_VECTOR_INDEX

    # Example (uncomment and adjust file paths):
    # df_202503 = pd.read_csv("202503.csv")
    # df_202601 = pd.read_csv("202601.csv")
    # courses_df = pd.concat([df_202503, df_202601], ignore_index=True)
    # COURSE_CATALOG = build_clean_course_catalog(courses_df)
    #
    # COURSE_VECTOR_INDEX = build_course_vector_index(COURSE_CATALOG)

    # For now, leave as no-op if you don't need RAG or scheduling with real data.
    pass


# if __name__ == "__main__":
    # Initialize course data if you want scheduling / RAG to be fully live
    # initialize_course_data()
    # interactive_chat()
