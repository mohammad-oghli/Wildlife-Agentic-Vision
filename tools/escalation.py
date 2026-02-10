def flag_at_risk(individual_id, confidence, explanation):
    print(f"\n[AT RISK]")
    print(f"Animal: {individual_id}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Explanation: {explanation}\n")

def monitor(individual_id):
    print(f"[MONITORING] {individual_id}")
