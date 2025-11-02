# ddi_app_with_condition_chat_alternatives.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import gradio as gr

# -------------------------
# Load datasets
# -------------------------
drugs = pd.read_csv("drugs.csv")
ddi = pd.read_csv("ddi_pairs.csv")

ddi = ddi.merge(drugs.add_suffix('_a'), left_on="drug_id_a", right_on="drug_id_a", how="left")
ddi = ddi.merge(drugs.add_suffix('_b'), left_on="drug_id_b", right_on="drug_id_b", how="left")

# -------------------------
# Encode + train model
# -------------------------
le_class = LabelEncoder()
le_class.fit(drugs["class"])

ddi["class_a_enc"] = le_class.transform(ddi["class_a"])
ddi["class_b_enc"] = le_class.transform(ddi["class_b"])
ddi["dose_mg_a"].fillna(0, inplace=True)
ddi["dose_mg_b"].fillna(0, inplace=True)

X = ddi[["dose_mg_a", "dose_mg_b", "class_a_enc", "class_b_enc"]]
y = ddi["severity"].astype(int)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# -------------------------
# Severity map
# -------------------------
severity_map = {
    0: {"label": "Safe", "effect": "No harmful interaction expected.", "advice": "Safe for most users."},
    1: {"label": "Mild", "effect": "May cause mild dizziness or nausea.", "advice": "Take after food and stay hydrated."},
    2: {"label": "Moderate", "effect": "Possible organ strain or absorption issues.", "advice": "Consult a doctor if symptoms persist."},
    3: {"label": "Severe", "effect": "Risk of toxicity or internal bleeding.", "advice": "Avoid combining unless prescribed."}
}

# -------------------------
# Condition map
# -------------------------
condition_risk_map = {
    "None": {"classes": [], "desc": "No special risk factors."},
    "Heart Disease": {"classes": ["NSAID", "Anticoagulant", "Antiplatelet"], "desc": "Heart patients should avoid NSAIDs and blood thinners."},
    "Diabetes": {"classes": ["Antidiabetic", "Corticosteroid"], "desc": "Steroids can raise blood sugar levels."},
    "Pregnancy": {"classes": ["NSAID", "Antibiotic", "Opioid"], "desc": "Certain antibiotics and NSAIDs are unsafe during pregnancy."},
    "Kidney Issues": {"classes": ["NSAID", "Diuretic"], "desc": "NSAIDs can worsen kidney function."},
    "Liver Issues": {"classes": ["Analgesic", "Statin"], "desc": "These can be hepatotoxic."}
}

# -------------------------
# Substitute map
# -------------------------
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

# -------------------------
# Predict Interaction
# -------------------------
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

    # Encode & predict
    c1 = le_class.transform([d1["class"]])[0]
    c2 = le_class.transform([d2["class"]])[0]
    sample = [[dose1_val, dose2_val, c1, c2]]
    pred = int(model.predict(sample)[0])

    # Apply condition effect
    cond = condition if condition in condition_risk_map else "None"
    cond_risk = condition_risk_map[cond]["classes"]
    cond_text = condition_risk_map[cond]["desc"]

    if d1["class"] in cond_risk or d2["class"] in cond_risk:
        pred = min(3, pred + 1)

    info = severity_map[pred]
    result = [
        f"Severity: {info['label']}",
        f"Effects: {info['effect']}",
        f"Advice: {info['advice']}",
        f"Condition applied: {cond} - {cond_text}"
    ]

    if warnings:
        result.append("\nDose Warnings:")
        result.extend(warnings)

    # Suggest safer alternatives
    suggestions = []
    if info["label"] in ["Moderate", "Severe"] or pred not in severity_map:
        if drug1 in drug_substitutes:
            suggestions.append(f"Alternative for {drug1}: {drug_substitutes[drug1]}")
        if drug2 in drug_substitutes:
            suggestions.append(f"Alternative for {drug2}: {drug_substitutes[drug2]}")
        if suggestions:
            result.append("\nRecommended Alternatives:")
            result.extend(suggestions)

    return "\n\n".join(result)

# -------------------------
# Chat Assistant
# -------------------------
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

        # Add alternative advice summary
        subs = []
        if d1 in drug_substitutes:
            subs.append(f"{d1} → {drug_substitutes[d1]}")
        if d2 in drug_substitutes:
            subs.append(f"{d2} → {drug_substitutes[d2]}")
        if subs:
            response += f"\n\nAlternative suggestions: {', '.join(subs)}"

    elif len(found) == 1:
        d = drugs.loc[drugs["name"] == found[0]].iloc[0]
        dose = d["recommended_dose_mg"] if "recommended_dose_mg" in d else "N/A"
        response = f"{d['name']} (Class: {d['class']}) — Recommended dose: {dose} mg."
    else:
        words = [w.capitalize() for w in msg.split() if w.isalpha()]
        unknown = [w for w in words if w not in drugs["name"].values]
        if unknown:
            warning_text = f"Warning: The following drugs are not found in the dataset: {', '.join(unknown)}"
            response = "Please provide valid drug names from the dataset."
        else:
            response = (
                "You can ask about:\n"
                "- Interactions (mention two drug names)\n"
                "- Doses (e.g., 'dose of Metformin')\n"
                "- Single drug details\n"
                "- Include patient condition for personalized result."
            )

    chat_history.append((message, response))
    return chat_history, chat_history, warning_text

# -------------------------
# UI
# -------------------------
drug_list = sorted(drugs["name"].tolist())
condition_list = list(condition_risk_map.keys())

with gr.Blocks() as demo:
    gr.Markdown("# Drug Interaction Advisor 💊")

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
