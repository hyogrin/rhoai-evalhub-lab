# NeMo Guardrails custom output parser for Kanana Safeguard 8B
# Mounted as config.py in the guardrail config directory (fixed filename)
#
# Kanana Safeguard outputs a single token:
#   <SAFE>       — content is safe
#   <UNSAFE-Sx>  — content violates category Sx (x = 1..7)
#
# Categories:
#   S1: Violent Crimes, S2: Non-Violent Crimes, S3: Sex-Related Crimes,
#   S4: Child Sexual Exploitation, S5: Suicide & Self-Harm,
#   S6: Hate, S7: Privacy Violation
#
# Reference: https://huggingface.co/kakaocorp/kanana-safeguard-8b


def init(rails):
    def kanana_parse_safety(response):
        """Parse Kanana Safeguard output: <SAFE> or <UNSAFE-Sx>."""
        cleaned = response.strip()
        upper = cleaned.upper()
        if "SAFE" in upper and "UNSAFE" not in upper:
            return [True]
        elif "UNSAFE" in upper:
            return [False, cleaned]
        # Default to unsafe if parsing fails
        return [False, "parse_error"]

    rails.register_output_parser(kanana_parse_safety, "kanana_parse_safety")
