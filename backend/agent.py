"""Provider-neutral explanation adapter. No credentials belong in frontend code."""
def explain(result):
    # Replace this function with a server-side provider call when selected.
    # Supply only trusted result data and verified knowledge records. Preserve
    # uncertainty; never infer edibility, toxins, disease, or beach occurrence.
    prefix = "DEMO ONLY — this label is a fixed example, not an image assessment. " if result["mode"] == "demo" else ""
    if result["task"] == "species":
        text = f"The {'example' if result['mode'] == 'demo' else 'predicted'} category is {result['label'].replace('_', ' ')} ({result['scientific_name']}). This prototype supports five species. Unsupported fish may still receive one of these labels."
    else:
        text = f"The {'example' if result['mode'] == 'demo' else 'predicted'} visual freshness category is {result['label'].replace('_', ' ')}. This is a dataset-defined eye-appearance category, not a measure of storage time or food safety."
    return {"provider": "local_template", "agent_connected": False,
            "text": prefix + text,
            "location_note": "No verified Sydney beach distribution record is connected.",
            "safety_note": "An image cannot confirm that a fish is healthy, toxin-free or safe to eat.",
            "sources": []}

