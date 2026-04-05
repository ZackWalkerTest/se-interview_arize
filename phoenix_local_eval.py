import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import phoenix as px

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROJECT_NAME = "travel-assistant-agent"

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------------------------------
# Connect to Phoenix
# --------------------------------------------------

phoenix = px.Client()

print("Connected to Phoenix")

# --------------------------------------------------
# Export spans
# --------------------------------------------------

print("Exporting spans...")

spans_df = phoenix.get_spans_dataframe(project_name=PROJECT_NAME)

print(f"Exported {len(spans_df)} spans")

# Save raw export
spans_df.to_csv("phoenix_spans_export.csv", index=False)

# --------------------------------------------------
# Inspect available columns
# --------------------------------------------------

print("\nAvailable columns:")
print(spans_df.columns)

# --------------------------------------------------
# Detect likely input/output columns
# --------------------------------------------------

input_col = None
output_col = None

for c in spans_df.columns:
    if "input" in c.lower():
        input_col = c
    if "output" in c.lower():
        output_col = c

print(f"\nDetected input column: {input_col}")
print(f"Detected output column: {output_col}")

# --------------------------------------------------
# Detect tool failures from span attributes
# --------------------------------------------------

ERROR_PATTERNS = [
    "401",
    "403",
    "500",
    "Unauthorized",
    "Forbidden",
    "Rate limit",
    "APIError",
]

def detect_tool_error(message):

    if message is None:
        return False

    message = str(message)

    return any(err in message for err in ERROR_PATTERNS)


attribute_column = "attributes.llm.input_messages"

if attribute_column in spans_df.columns:

    spans_df["tool_error"] = spans_df[attribute_column].apply(
        detect_tool_error
    )

else:

    spans_df["tool_error"] = False

print("\nTool error detection results:")
print(spans_df["tool_error"].value_counts())

# --------------------------------------------------
# Initialize frustration label
# --------------------------------------------------

spans_df["frustration_label"] = "NOT_FRUSTRATED"

# Automatically label tool failures as frustration
spans_df.loc[
    spans_df["tool_error"] == True,
    "frustration_label"
] = "FRUSTRATED"

# --------------------------------------------------
# LLM-as-a-Judge prompt
# --------------------------------------------------

PROMPT_TEMPLATE = """
You are evaluating a travel assistant interaction.

User message:
{input}

Assistant response:
{output}

Determine if the user appears frustrated.

Signs of frustration include:
- Tool errors or failures
- Agent using language indicating failure (e.g. "Sorry, I can't find that information." or "I encountered an issue.")
- Agent falling back to a regular search instead of using tools when it seems appropriate
- task failure

Signs of NOT being frustrated include:
- Agent responding with a helpful answer, even if it's not perfect
- Agent invoking tools appropriately with no API errors and providing an answer based on the tool results

If the interaction clearly indicates frustration, respond:

FRUSTRATED

Otherwise respond:

NOT_FRUSTRATED

Respond with only one label.
"""

# --------------------------------------------------
# Run LLM judge on remaining spans
# --------------------------------------------------

print("\nRunning LLM frustration evaluation...")

for idx, row in spans_df.iterrows():

    # Skip spans already flagged by tool failure
    if row["frustration_label"] == "FRUSTRATED":
        continue

    user_input = str(row.get(input_col, ""))
    assistant_output = str(row.get(output_col, ""))

    prompt = PROMPT_TEMPLATE.format(
        input=user_input,
        output=assistant_output
    )

    try:

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        label = response.choices[0].message.content.strip()

        if "FRUSTRATED" in label:
            spans_df.loc[idx, "frustration_label"] = "FRUSTRATED"

    except Exception as e:

        print(f"LLM evaluation failed on row {idx}: {e}")

# --------------------------------------------------
# Save evaluation results
# --------------------------------------------------

spans_df.to_csv("phoenix_frustration_eval_results.csv", index=False)

print("\nEvaluation complete")

# --------------------------------------------------
# Show frustrated interactions
# --------------------------------------------------

frustrated = spans_df[
    spans_df["frustration_label"] == "FRUSTRATED"
]

print(f"\nDetected {len(frustrated)} frustrated interactions\n")

if len(frustrated) > 0:
    print(frustrated.head())

# --------------------------------------------------
# Final output
# --------------------------------------------------

print("\nSaved files:")
print("phoenix_spans_export.csv")
print("phoenix_frustration_eval_results.csv")

# Attach frustration labels back to Phoenix spans as annotations

from phoenix.client import Client

px_client = Client()

annotations = []

for _, row in spans_df.iterrows():

    span_id = row.get("span_id") or row.get("context.span_id")

    if span_id is None:
        continue

    annotations.append(
        {
            "name": "user_frustration",
            "span_id": span_id,
            "annotator_kind": "LLM",
            "result": {
                "label": row["frustration_label"],
                "score": 1.0 if row["frustration_label"] == "FRUSTRATED" else 0.0
            },
            "metadata": {
                "source": "local_eval_script",
                "model": "gpt-4o"
            }
        }
    )

print(f"\nUploading {len(annotations)} annotations to Phoenix...")

px_client.spans.log_span_annotations(
    span_annotations=annotations,
    sync=False
)

print("Frustration annotations uploaded successfully.")

# --------------------------------------------------
# Upload tool_error annotations to Phoenix
# --------------------------------------------------

tool_error_spans = spans_df[spans_df["tool_error"] == True]

if len(tool_error_spans) == 0:
    print("No tool errors detected, nothing to upload.")
else:
    tool_error_annotations = []

    for _, row in tool_error_spans.iterrows():
        span_id = str(row.get("span_id") or row.get("context.span_id"))
        if not span_id:
            continue

        tool_error_annotations.append({
            "span_id": span_id,
            "name": "tool_error",
            "annotator_kind": "CODE",  # must be LLM, CODE, or HUMAN
            "result": {"label": "error"},  # plain string label in dict
            "metadata": {"source": "local_eval_script"}
        })

    print(f"\nUploading {len(tool_error_annotations)} tool error annotations to Phoenix...")

    try:
        px_client.spans.log_span_annotations(
            span_annotations=tool_error_annotations,
            sync=False
        )
        print("Tool error annotations uploaded successfully.")
    except Exception as e:
        print(f"Upload failed: {e}")
        for anno in tool_error_annotations:
            print(anno)