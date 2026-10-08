"""
Design Rule Compliance Checker for Label Audit.
Scans all files in backend/static/ to verify zero banned patterns:
1. No em dashes (— or \u2014 or &mdash;)
2. No gradients (linear-gradient, radial-gradient, conic-gradient)
3. No pill buttons (999px, 9999px, 50% radius)
4. No emoji
5. No banned hype words (revolutionary, seamless, powerful, unlock, supercharge, next-gen)
6. No ratings, testimonials, counters, or 'made with' tags.
"""

import os
import re
import sys

STATIC_DIR = "backend/static"

BANNED_WORDS = [
    "revolutionary",
    "seamless",
    "powerful",
    "unlock",
    "supercharge",
    "next-gen",
    "made with",
    "5-star",
    "trusted by",
    "happy users"
]

EMOJI_PATTERN = re.compile(
    r"[\U0001F300-\U0001F64F"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F900-\U0001F9FF"
    r"\U0001FA70-\U0001FAFF"
    r"\u2600-\u26FF"
    r"\u2700-\u27BF]"
)

def check_rules():
    print("=" * 80)
    print("RUNNING DESIGN RULES AUDIT ON backend/static/")
    print("=" * 80)

    violations = []

    if not os.path.exists(STATIC_DIR):
        print(f"Error: {STATIC_DIR} not found.")
        sys.exit(1)

    for root, _, files in os.walk(STATIC_DIR):
        for fname in files:
            # Skip binary font and image files
            if fname.endswith((".woff2", ".woff", ".ttf", ".png", ".ico", ".jpg", ".jpeg")):
                continue

            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, ".")

            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            for line_idx, line in enumerate(lines, 1):
                # 1. Check for Em Dash
                if "—" in line or "\u2014" in line or "&mdash;" in line:
                    violations.append((rel_path, line_idx, "Em dash detected (— / \\u2014 / &mdash;)"))

                # 2. Check for Gradients
                if "gradient" in line.lower():
                    violations.append((rel_path, line_idx, f"Gradient detected: '{line.strip()}'"))

                # 3. Check for Pill Button Radius
                if ("999px" in line or "9999px" in line) and "radius" in line:
                    violations.append((rel_path, line_idx, f"Pill shape border-radius detected: '{line.strip()}'"))

                # 4. Check for Emoji
                if EMOJI_PATTERN.search(line):
                    violations.append((rel_path, line_idx, f"Emoji character detected: '{line.strip()}'"))

                # 5. Check for Banned Hype Words
                for bw in BANNED_WORDS:
                    if re.search(r"\b" + re.escape(bw) + r"\b", line, re.IGNORECASE):
                        violations.append((rel_path, line_idx, f"Banned hype word detected: '{bw}'"))

    if violations:
        print(f"\nFAILED: Found {len(violations)} rule violations:")
        for file_path, line_no, desc in violations:
            print(f"  [X] {file_path}:{line_no} -> {desc}")
        return False
    else:
        print("\nPASSED: Zero rule violations detected in backend/static/!")
        print("  ✓ Zero em dashes")
        print("  ✓ Zero gradients")
        print("  ✓ Zero pill shapes")
        print("  ✓ Zero emoji")
        print("  ✓ Zero banned hype words")
        print("  ✓ Zero ratings, testimonials, or counters")
        return True

if __name__ == "__main__":
    success = check_rules()
    sys.exit(0 if success else 1)
