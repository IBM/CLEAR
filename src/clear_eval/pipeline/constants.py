IDENTIFIED_SHORTCOMING_COL = "identified_shortcomings"
EVALUATION_TEXT_COL = "evaluation_text"
EVALUATION_SUMMARY_COL = "evaluation_summary"
SCORE_COL = "score"
ERROR_COL = "error"
SHORTCOMING_PREFIX ='shortcoming_'
GENERATION_FILE_PREFIX = "dataset_with_generations"
EVALUATION_FILE_PREFIX_NO_SUMMARIES = "per_record_evaluations_no_summaries"
EVALUATION_FILE_PREFIX_WITH_SUMMARIES = "per_record_evaluations"
SHORTCOMING_LIST_FILE_PREFIX = "shortcoming_list"
MAPPING_FILE_PREFIX = "mapping_results"
MAPPING_NO_ISSUES = "NO_ISSUES"
ANALYSIS_SKIPPED = "Analysis Skipped"
DEFAULT_ISSUES_FORMAT_MODE = "shortcomings"
DEFAULT_SPARC_TRACK = "fast_track"

# SPARC-specific per-row columns produced by the tool-call use case. They are
# carried through analysis CSVs unchanged so downstream aggregation
# (build_json_results) and dashboards can read per-row judgments without
# re-running SPARC.
_SPARC_COLUMNS_PASSTHROUGH = (
    "sparc_decision",
    "sparc_score_1_to_5",
    "sparc_recommendations",
)