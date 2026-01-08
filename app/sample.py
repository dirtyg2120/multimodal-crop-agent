from app.agent.core import DiagnosisResult


output = DiagnosisResult(
    reasoning="",
    overall_health_status = "Severe Infestation",
    identified_pathogens = [
        "Late blight",
        "Algal Leaf Spot",
        "Bacterial blight"
    ],
    severity_level = "High Severity",
    infection_ratio = 0.857,
    recommended_actions = [
        "Manual lookup failed; recommendation based on general agronomic principles."
    ],
    required_pesticides = [
        "Copper Fungicide"
    ]
)