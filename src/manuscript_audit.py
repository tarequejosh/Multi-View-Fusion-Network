import re

def audit_manuscript():
    with open('paper/manuscript.md', 'r', encoding='utf-8') as f:
        content = f.read()
        
    checks = []
    
    # Required metrics
    checks.append(("Has B3 Accuracy 99.30", "99.30" in content or "0.9930" in content))
    checks.append(("Has B3 F1 99.38", "99.38" in content or "0.9938" in content))
    checks.append(("Has B3 AUC 99.93", "99.93" in content or "0.9993" in content))
    checks.append(("Has B3 Dice 87.93", "87.93" in content or "0.8793" in content))
    checks.append(("Has B3 HD95 3.44", "3.44" in content))
    
    # Old erroneous metrics MUST NOT be present
    checks.append(("No old Accuracy 99.33", "99.33" not in content))
    checks.append(("No old Dice 87.64", "87.64" not in content))
    
    # Forbidden terms MUST NOT be present
    forbidden = [r"\bSOTA\b", r"\bfirst\b", r"clinical validation", r"unseen clinical hardware", r"\bproves\b", r"caused by domain shift"]
    for term in forbidden:
        clean_name = term.replace(r"\b", "")
        # Search using regex
        match = re.search(term, content, re.IGNORECASE)
        checks.append((f"No '{clean_name}'", match is None))
        
    # Formatting checks
    # Make sure we don't present 59.54% Dice without explaining empty mask
    has_dice = "59.54%" in content
    has_explanation = "empty-mask" in content.lower() and "dice convention" in content.lower()
    if has_dice:
        checks.append(("59.54% Dice explained", has_explanation))
        
    # Write report
    report_lines = ["# Manuscript Consistency Audit\n"]
    all_passed = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        report_lines.append(f"- **{status}**: {name}")
        if not passed:
            all_passed = True # Wait, bug in logic, fix below:
            all_passed = False
            
    report_lines.append("\n**Overall Status**: " + ("PASS" if all_passed else "FAIL"))
    
    with open('reports/manuscript_external_validation_consistency_audit.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    print("Audit Complete. All Passed:", all_passed)

if __name__ == "__main__":
    audit_manuscript()
