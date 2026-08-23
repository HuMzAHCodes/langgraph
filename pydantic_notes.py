"""
PYDANTIC NOTES — a single reference file covering everything you've used
across the LangChain/LangGraph labs: BaseModel, Field, Literal, Optional,
Annotated, and how they combine to define validated schemas.

Nothing here needs to be run against an LLM — these are pure Pydantic
concepts. Run this file directly (python pydantic_notes.py) to see every
concept demonstrated with real output.
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Literal, Annotated


# ─────────────────────────────────────────────────────────────────────────────
# 1. BaseModel — the foundation
# ─────────────────────────────────────────────────────────────────────────────
# A BaseModel is a class that VALIDATES data at construction time, unlike a
# plain Python class or a TypedDict. Each attribute is a field: `name: type`.
# Subclassing BaseModel is what turns these annotations into live validation
# rules -- get the type wrong, and Pydantic raises a ValidationError instead
# of silently letting bad data through.

class Student(BaseModel):
    name: str          # required -- no default, so this MUST be provided
    age: int            # required


# ─────────────────────────────────────────────────────────────────────────────
# 2. Required vs Optional fields
# ─────────────────────────────────────────────────────────────────────────────
# Three distinct cases:
#   field: str                      -> REQUIRED (no default)
#   field: str = 'default value'    -> OPTIONAL, defaults if omitted
#   field: Optional[str] = None     -> the TYPE allows str OR None, AND it's
#                                       omittable because of the `= None` default
#
# IMPORTANT: Optional[X] on its own only changes the TYPE (X or None). It does
# NOT make the field skippable by itself -- you still need a default (usually
# None) for that. A field typed Optional[int] with NO default is still
# required (you'd just be allowed to explicitly pass None for it).

class StudentWithDefaults(BaseModel):
    name: str = 'nitish'                 # optional, defaults to 'nitish'
    age: Optional[int] = None            # optional, may be int or None
    email: str                           # required, no default


# ─────────────────────────────────────────────────────────────────────────────
# 3. Literal — restricting a value to an exact fixed set
# ─────────────────────────────────────────────────────────────────────────────
# Literal["a", "b", "c"] means the field can ONLY be one of those exact
# values -- nothing else is valid, not even something semantically close.
# This is different from an Enum in usage but does the same job: a hard,
# closed set of allowed values.
#
# You used this constantly in the LangGraph labs:
#   sentiment: Literal["positive", "negative"]
#   issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"]
#   tone: Literal["angry", "frustrated", "disappointed", "calm"]
#   urgency: Literal["low", "medium", "high"]
#
# CLASSIC GOTCHA: if your field's natural-language DESCRIPTION mentions a
# value that ISN'T in the Literal, the Literal always wins. E.g. a
# description saying "positive, negative, or neutral" but the Literal is
# only Literal["positive", "negative"] -- "neutral" is IMPOSSIBLE to produce,
# no matter what the description implies. The Literal is what's enforced;
# the description is only an instruction/hint to whoever (or whatever LLM)
# is filling the field in.

class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the review"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Field() — constraints and metadata on a single field
# ─────────────────────────────────────────────────────────────────────────────
# Field() lets you attach VALIDATION RULES and DOCUMENTATION to a field, on
# top of its type. Common arguments:
#
#   default=...       value used if the field is omitted (alternative to `= x`)
#   description=...   pure metadata/documentation -- does NOT affect validation,
#                      but if this schema is handed to an LLM (e.g. via
#                      with_structured_output or a PydanticOutputParser), this
#                      description is what the LLM reads to know what to put here
#   gt / lt            value must be greater-than / less-than (exclusive)
#   ge / le            value must be greater-or-equal / less-or-equal (inclusive)
#   min_length /
#   max_length          length constraints for strings, lists, etc.
#   pattern             a regex the string must match
#
# Field() is what makes Pydantic more powerful than a bare type hint: the
# TYPE says "this must be a float"; Field(gt=0, lt=10) says "and it must
# specifically be between 0 and 10."

class ScoredEvaluation(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)
    cgpa: float = Field(gt=0, lt=10, default=5, description="A decimal cgpa value")


# ─────────────────────────────────────────────────────────────────────────────
# 5. EmailStr — a specialised validated type
# ─────────────────────────────────────────────────────────────────────────────
# EmailStr is Pydantic's built-in type that validates a string IS a
# well-formed email address, not just any string. Needs the extra package
# `email-validator` installed (pip install email-validator), or the class
# definition itself raises an ImportError.

class ContactInfo(BaseModel):
    email: EmailStr   # 'abc@gmail.com' passes; 'abc' fails validation


# ─────────────────────────────────────────────────────────────────────────────
# 6. Annotated — attaching metadata alongside a type
# ─────────────────────────────────────────────────────────────────────────────
# Annotated[SomeType, extra_stuff] lets you attach EXTRA INFORMATION to a
# type without changing the type itself. Pydantic reads that extra info in
# different ways depending on what "extra_stuff" is.
#
# Two DIFFERENT uses of Annotated you've encountered across the labs:
#
# (a) Annotated with a plain description string -- the OLDER/alternate
#     TypedDict-style way of documenting a field (equivalent in spirit to
#     Field(description=...), but for TypedDict, which has no Field()):
#
#         from typing import TypedDict
#         class Review(TypedDict):
#             summary: Annotated[str, "A brief summary of the review"]
#
#     Here "A brief summary of the review" is JUST a description string,
#     read by whatever consumes the schema (e.g. an LLM via
#     with_structured_output). TypedDict itself does zero validation --
#     Annotated here is documentation only, not enforcement.
#
# (b) Annotated with a REDUCER function -- this is the LangGraph use you
#     saw in the UPSC essay workflow, and it means something completely
#     different: it tells LangGraph HOW TO COMBINE old and new values for
#     that state key, instead of just overwriting:
#
#         import operator
#         class UPSCState(TypedDict):
#             individual_scores: Annotated[list[int], operator.add]
#
#     Here operator.add is NOT a description -- it's a REDUCER FUNCTION.
#     When multiple graph nodes each return a value for individual_scores,
#     LangGraph calls operator.add(existing_list, new_list) to MERGE them
#     (list concatenation) instead of the last writer silently overwriting
#     everyone else's contribution.
#
# The pattern in both cases is IDENTICAL in syntax -- Annotated[Type, X] --
# but WHAT X does depends entirely on WHO is reading the annotation:
#   - a plain string  -> documentation/description (TypedDict schema case)
#   - a function       -> a reducer telling LangGraph how to merge state
#                          (only meaningful inside a LangGraph TypedKey state)

import operator
from typing import TypedDict

class ReviewFeedback(TypedDict):
    # (a) Annotated as description -- just documentation for an LLM schema
    summary: Annotated[str, "A one-line summary of the review"]

class UPSCState(TypedDict):
    # (b) Annotated as reducer -- tells LangGraph to CONCATENATE lists
    # written by different parallel nodes, instead of overwriting
    individual_scores: Annotated[list[int], operator.add]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Nested schemas — a BaseModel as a Literal-constrained multi-field record
# ─────────────────────────────────────────────────────────────────────────────
# A realistic schema combines everything above: some required fields, some
# Literal-constrained fields, Field() descriptions throughout, and often
# Optional fields for data that might not always be present.

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(
        description="The category of issue mentioned in the review"
    )
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(
        description="The emotional tone expressed by the user"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="How urgent or critical the issue appears to be"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Any extra free-text notes -- may be absent"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Getting data back OUT of a model
# ─────────────────────────────────────────────────────────────────────────────
#   model.model_dump()        -> plain Python dict (Pydantic v2's official way;
#                                 handles nested models, exclude/include, etc.)
#   model.model_dump_json()   -> a JSON STRING (for APIs, files, logging)
#   dict(model)                -> older/looser shallow dict conversion
#
# (Pydantic v1 used .dict() / .json(); v2 renamed these to model_dump /
# model_dump_json -- if you see the old names in a tutorial, that's v1 code.)


# ─────────────────────────────────────────────────────────────────────────────
# DEMONSTRATION — run this file to see everything above in action
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("── 1. BaseModel + required fields ──")
    s = Student(name="Amit", age=21)
    print(s)

    print("\n── 2. Optional fields with defaults ──")
    s2 = StudentWithDefaults(email="amit@example.com")   # name & age omitted
    print(s2)  # name defaults to 'nitish', age defaults to None

    print("\n── 3. Literal — only exact values allowed ──")
    sent = SentimentSchema(sentiment="positive")
    print(sent)
    try:
        SentimentSchema(sentiment="neutral")   # NOT in the Literal -> fails
    except Exception as e:
        print("Expected failure for 'neutral':", type(e).__name__)

    print("\n── 4. Field() constraints (ge/le, gt/lt) ──")
    ev = ScoredEvaluation(feedback="Well argued essay.", score=8)
    print(ev)   # cgpa uses its Field default of 5
    try:
        ScoredEvaluation(feedback="Bad score test", score=15)  # score > 10 -> fails
    except Exception as e:
        print("Expected failure for score=15:", type(e).__name__)

    print("\n── 5. EmailStr validation ──")
    contact = ContactInfo(email="student@example.com")
    print(contact)
    try:
        ContactInfo(email="not-an-email")   # fails format validation
    except Exception as e:
        print("Expected failure for bad email:", type(e).__name__)

    print("\n── 6a. Annotated as description (TypedDict, no enforcement) ──")
    review_note: ReviewFeedback = {"summary": "Clear and concise."}
    print(review_note)  # TypedDict does NOT validate -- this is just a dict

    print("\n── 6b. Annotated as reducer (conceptual — real merging happens inside LangGraph) ──")
    # Outside of LangGraph, operator.add just merges two lists directly:
    merged = operator.add([7], [6])
    merged = operator.add(merged, [8])
    print("scores merged like LangGraph's reducer would:", merged)

    print("\n── 7. Nested Literal-based schema ──")
    diag = DiagnosisSchema(issue_type="Bug", tone="frustrated", urgency="high")
    print(diag)

    print("\n── 8. Exporting model data ──")
    print("model_dump():     ", diag.model_dump())
    print("model_dump_json():", diag.model_dump_json())
