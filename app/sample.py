from app.agent.core import DiagnosisResult


output = DiagnosisResult(
    overall_health_status = "Severe Infestation",
    identified_pathogens = [
        "Late blight",
        "Algal Leaf Spot",
        "Bacterial blight"
    ],
    severity_level = "High Severity",
    infection_ratio = 0.857,
    recommended_actions = [
        "Manual lookup failed; recommendation based on general agronomic principles.",
        "Immediately remove and destroy all severely infected leaves and plant parts to reduce inoculum.",
        "Improve air circulation around the plants by appropriate spacing and pruning.",
        "Apply a broad-spectrum copper-based fungicide/bactericide to manage both fungal (Late Blight, Algal Leaf Spot) and bacterial diseases (Bacterial Blight).",
        "Monitor plants daily for new or worsening symptoms."
    ],
    required_pesticides = [
        "Copper Fungicide"
    ]
)