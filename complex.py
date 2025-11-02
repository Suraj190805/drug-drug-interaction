# ddi_app_final.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import gradio as gr
import os

# ======================================================
# STEP 1 — Load datasets
# ======================================================
if not os.path.exists("drugs_enhanced.csv"):
    raise FileNotFoundError("Missing file: drugs_enhanced.csv")
if not os.path.exists("ddi_pairs_enhanced.csv"):
    raise FileNotFoundError("Missing file: ddi_pairs_enhanced.csv")

drugs = pd.read_csv("drugs_enhanced.csv")
ddi = pd.read_csv("ddi_pairs_enhanced.csv")

# ======================================================
# STEP 2 — Ensure proper columns
# ======================================================
for col in ["Drug 1", "Drug 2", "drug_id_a", "drug_id_b"]:
    if col not in ddi.columns:
        raise KeyError(f"Missing column '{col}' in ddi_pairs_enhanced.csv")

if "name" not in drugs.columns:
    raise KeyError("Missing 'name' column in drugs_enhanced.csv")

# ======================================================
# STEP 3 — Merge drug info
# ======================================================
ddi = ddi.merge(drugs.add_suffix('_a'), left_on="drug_id_a", right_on="drug_id_a", how="left")
ddi = ddi.merge(drugs.add_suffix('_b'), left_on="drug_id_b", right_on="drug_id_b", how="left")

# ======================================================
# STEP 4 — Add severity (rule-based)
# ======================================================
severe_drugs = ["Warfarin", "Prednisone", "Ibuprofen", "Aspirin", "Naproxen", "Diclofenac", "Ciprofloxacin"]
moderate_drugs = ["Omeprazole", "Metformin", "Paracetamol", "Amoxicillin", "Atorvastatin"]

def assign_severity(row):
    d1, d2 = str(row["Drug 1"]), str(row["Drug 2"])
    if any(drug in (d1, d2) for drug in severe_drugs):
        return 3
    elif any(drug in (d1, d2) for drug in moderate_drugs):
        return 2
    elif d1[0] == d2[0]:
        return 1
    else:
        return 0

if "severity" not in ddi.columns:
    ddi["severity"] = ddi.apply(assign_severity, axis=1)

# ======================================================
# STEP 5 — Train model
# ======================================================
ddi["recommended_dose_mg_a"].fillna(0, inplace=True)
ddi["recommended_dose_mg_b"].fillna(0, inplace=True)

X = ddi[["recommended_dose_mg_a", "recommended_dose_mg_b"]]
y = ddi["severity"].astype(int)

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# ======================================================
# STEP 6 — Severity map
# ======================================================
severity_map = {
    0: {"label": "Safe", "effect": "No harmful interaction expected.", "advice": "Safe for most users."},
    1: {"label": "Mild", "effect": "May cause mild dizziness or nausea.", "advice": "Take after food and stay hydrated."},
    2: {"label": "Moderate", "effect": "Possible organ strain or absorption issues.", "advice": "Consult a doctor if symptoms persist."},
    3: {"label": "Severe", "effect": "Risk of toxicity or internal bleeding.", "advice": "Avoid combining unless prescribed."}
}

# ======================================================
# STEP 7 — Condition map
# ======================================================
condition_risk_map = {
    "None": {"desc": "No special risk factors."},
    "Heart Disease": {"desc": "Heart patients should avoid blood thinners and NSAIDs."},
    "Diabetes": {"desc": "Steroids and some antibiotics can raise blood sugar levels."},
    "Pregnancy": {"desc": "Certain antibiotics and NSAIDs are unsafe during pregnancy."},
    "Kidney Issues": {"desc": "NSAIDs can worsen kidney function."},
    "Liver Issues": {"desc": "Certain drugs can cause liver strain."}
}

# ======================================================
# STEP 8 — Substitute map
# ======================================================
drug_substitutes = {
    "Ibuprofen": "Paracetamol",
    "Aspirin": "Paracetamol",
    "Naproxen": "Paracetamol",
    "Diclofenac": "Paracetamol",
    "Warfarin": "Apixaban",
    "Omeprazole": "Pantoprazole",
    "Ciprofloxacin": "Amoxicillin",
    "Prednisone": "Hydrocortisone"
}

# ======================================================
# STEP 9 — Predict interaction
# ======================================================
def predict_interaction(drug1, dose1, drug2, dose2, condition="None"):
    missing = [d for d in [drug1, drug2] if d not in drugs["name"].values]
    if missing:
        return f"Warning: The following drugs are not found in the dataset: {', '.join(missing)}"

    if not drug1 or not drug2:
        return "Please select both drugs."
    if drug1 == drug2:
        return "Same drug selected twice — no interaction prediction."

    d1 = drugs[drugs["name"] == drug1].iloc[0]
    d2 = drugs[drugs["name"] == drug2].iloc[0]

    try:
        dose1_val = float(dose1)
        dose2_val = float(dose2)
    except:
        return "Please enter valid numeric doses."

    warnings = []
    if "recommended_dose_mg" in drugs.columns:
        if dose1_val > d1["recommended_dose_mg"]:
            warnings.append(f"{drug1} dose ({dose1_val} mg) exceeds recommended {d1['recommended_dose_mg']} mg.")
        if dose2_val > d2["recommended_dose_mg"]:
            warnings.append(f"{drug2} dose ({dose2_val} mg) exceeds recommended {d2['recommended_dose_mg']} mg.")

    sample = [[dose1_val, dose2_val]]
    pred = int(model.predict(sample)[0])
    pred = min(3, pred + (1 if condition != "None" else 0))

    info = severity_map[pred]
    result = [
        f"Severity: {info['label']}",
        f"Effects: {info['effect']}",
        f"Advice: {info['advice']}",
        f"Condition: {condition} — {condition_risk_map[condition]['desc']}"
    ]

    if warnings:
        result.append("\nDose Warnings:")
        result.extend(warnings)

    suggestions = []
    if pred >= 2:
        if drug1 in drug_substitutes:
            suggestions.append(f"Alternative for {drug1}: {drug_substitutes[drug1]}")
        if drug2 in drug_substitutes:
            suggestions.append(f"Alternative for {drug2}: {drug_substitutes[drug2]}")
        if suggestions:
            result.append("\nRecommended Alternatives:")
            result.extend(suggestions)

    return "\n\n".join(result)

# ======================================================
# STEP 10 — Chat Assistant
# ======================================================
def chat_assistant(message, chat_history, chat_condition):
    msg = message.lower().strip()
    found = [d for d in drugs["name"] if d.lower() in msg]
    found = list(dict.fromkeys(found))
    warning_text = ""

    if len(found) == 2:
        d1, d2 = found
        response = predict_interaction(
            d1,
            drugs.loc[drugs["name"] == d1, "recommended_dose_mg"].iloc[0],
            d2,
            drugs.loc[drugs["name"] == d2, "recommended_dose_mg"].iloc[0],
            chat_condition,
        )

    elif len(found) == 1:
        d = drugs.loc[drugs["name"] == found[0]].iloc[0]
        dose = d["recommended_dose_mg"] if "recommended_dose_mg" in d else "N/A"
        response = f"{d['name']} — Recommended dose: {dose} mg."
    else:
        warning_text = "Please mention one or two valid drug names from the dataset."
        response = (
            "You can ask about:\n"
            "- Interactions (mention two drug names)\n"
            "- Doses (e.g., 'dose of Metformin')\n"
            "- Single drug details\n"
            "- Include patient condition for personalized result."
        )

    chat_history.append((message, response))
    return chat_history, chat_history, warning_text

# ======================================================
# STEP 11 — UI
# ======================================================
drug_list = sorted(drugs["name"].tolist())
condition_list = list(condition_risk_map.keys())

with gr.Blocks() as demo:
    gr.Markdown("# 💊 Drug Interaction Advisor — Enhanced")

    with gr.Tab("Check Interaction"):
        with gr.Row():
            with gr.Column():
                drug1 = gr.Dropdown(drug_list, label="First Drug")
                dose1 = gr.Textbox(label="Dose 1 (mg)")
            with gr.Column():
                drug2 = gr.Dropdown(drug_list, label="Second Drug")
                dose2 = gr.Textbox(label="Dose 2 (mg)")
        condition = gr.Dropdown(condition_list, label="Patient Condition", value="None")
        predict_btn = gr.Button("Predict Interaction")
        output = gr.Textbox(label="Result", lines=12)
        predict_btn.click(predict_interaction, inputs=[drug1, dose1, drug2, dose2, condition], outputs=output)

    with gr.Tab("Chat Assistant"):
        warning_box = gr.Textbox(label="Warnings", value="", interactive=False)
        chatbot = gr.Chatbot(label="Assistant Chat")
        chat_condition = gr.Dropdown(condition_list, label="Patient Condition", value="None")
        msg = gr.Textbox(label="Message", placeholder="Ask about drug interactions or doses...")
        send = gr.Button("Send")
        clear = gr.Button("Clear Chat")

        send.click(chat_assistant, inputs=[msg, chatbot, chat_condition],
                   outputs=[chatbot, chatbot, warning_box])
        clear.click(lambda: ([], "", ""), None, outputs=[chatbot, warning_box])

if __name__ == "__main__":
    demo.launch()
