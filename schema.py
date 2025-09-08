# rag/schema.py

# JSON schema used by the vector-store path (RegulAIte)
# The pipeline will pass this as a JSON schema (or embed it in the prompt)
# so the model returns a single JSON object with these fields.

REG_ANSWER_SCHEMA = {
    "name": "RegulAIteAnswer",
    "schema": {
        "type": "object",
        "properties": {
            # Which frameworks the model considered for the query
            "query_frameworks": {"type": "array", "items": {"type": "string"}},

            # Findings per source family (each section can include snippets/quotes)
            "per_source": {
                "type": "object",
                "properties": {
                    "IFRS": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},  # e.g., "found" / "not_found" / "partial"
                            "notes": {"type": "string"},
                            "snippets": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "quote": {"type": "string"},
                                        "ref": {"type": "string"},        # para/section/clause
                                        "source": {"type": "string"},     # filename or doc name
                                        "page": {"type": ["integer", "string", "null"]},
                                    },
                                    "required": ["quote"]
                                }
                            }
                        },
                        "required": ["status"]
                    },
                    # Reuse the same shape for the other sources
                    "AAOIFI": {"$ref": "#/schema/properties/per_source/properties/IFRS"},
                    "CBB": {"$ref": "#/schema/properties/per_source/properties/IFRS"},
                    "InternalPolicy": {"$ref": "#/schema/properties/per_source/properties/IFRS"},
                }
            },

            # Cross-source comparison
            "comparative_analysis": {
                "type": "object",
                "properties": {
                    "alignment": {"type": "array", "items": {"type": "string"}},
                    "differences": {"type": "array", "items": {"type": "string"}},
                    "conflicts": {"type": "array", "items": {"type": "string"}}
                }
            },

            # Actionable recommendation
            "recommendation": {
                "type": "object",
                "properties": {
                    "guidance": {"type": "string"},
                    "follow_standard": {"type": "string"},          # e.g., "IFRS" / "AAOIFI" / "CBB" / "InternalPolicy"
                    "policy_update_needed": {"type": ["boolean", "null"]},
                    "policy_update_notes": {"type": "string"}
                }
            },

            # Optional tail sections
            "general_knowledge": {"type": "string"},
            "gaps_or_next_steps": {"type": "array", "items": {"type": "string"}},

            # Flat citation list (optional; separate from per_source.snippets)
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "page": {"type": ["integer", "string", "null"]},
                        "quote": {"type": "string"}
                    },
                    "required": ["source", "quote"]
                }
            }
        },
        # minimally require per_source so the UI has something to render
        "required": ["per_source"]
    }
}
