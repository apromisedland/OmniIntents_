"""The eight general and 21 specific categories in manuscript Table 1."""

GENERAL_TO_SPECIFIC = {
    "Request Guide": ("Teaching", "Error Correction", "Shopping Guide", "Location Guide"),
    "Request Companion": ("Interact with Companion", "Listen to Companion", "Watch with Companion"),
    "Request Collaborator": ("Request Collaborator",),
    "Request Search/Recognition": ("Search/Recognize Physical Items", "Search/Recognize Digital Information"),
    "Request Remote Control": ("Request Remote Control",),
    "Request Maintenance": ("Repair", "Check", "Replace", "Clean"),
    "Request Emergency Help": ("Top-priority Actions", "Rescue"),
    "Request Information Management": ("Save Documents", "Share Documents", "Edit Documents", "Generate Documents"),
}
SPECIFIC_TO_GENERAL = {
    specific: general
    for general, categories in GENERAL_TO_SPECIFIC.items()
    for specific in categories
}
GENERAL_LABELS = tuple(GENERAL_TO_SPECIFIC)
SPECIFIC_LABELS = tuple(SPECIFIC_TO_GENERAL)


class IntentTaxonomy:
    def general_for(self, specific: str) -> str:
        from ..errors import ValidationError

        if specific not in SPECIFIC_TO_GENERAL:
            raise ValidationError(f"Unknown specific intent: {specific!r}")
        return SPECIFIC_TO_GENERAL[specific]

    def to_prompt_list(self) -> str:
        return "\n".join(
            f"{general}: {', '.join(specific)}"
            for general, specific in GENERAL_TO_SPECIFIC.items()
        )
