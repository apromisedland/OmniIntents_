"""Intent-conditioned prompting; no claim of modifying neural attention weights."""

from ..types import Intent, StructuredText


class IntentFocusedAttentionShifter:
    def shift(self, structured: StructuredText, intent: Intent) -> str:
        label = intent.specific_label
        if label.startswith("Search/Recognize"):
            detail = "Read the selected object's visible title, label, identity and distinguishing details."
        elif label == "Location Guide":
            detail = "Attend to the requested destination, landmarks and route constraints for visual guidance."
        elif label in {"Request Collaborator", "Repair", "Replace", "Clean", "Rescue", "Request Remote Control"}:
            detail = "Attend to the target, relevant spatial relations, obstacles, tools and task constraints."
        elif label in {"Interact with Companion", "Listen to Companion", "Watch with Companion"}:
            detail = "Attend to the requested social interaction and distinguish gaze distractions from the user's request."
        else:
            detail = "Attend to the selected intent's target and the details needed to fulfill that request."
        evidence = "Reinspect the provided original images." if structured.image_paths else "Use the supplied pre-extracted features."
        return f"Selected intent: {intent.description}. {evidence} {detail} Do not invent unseen details."
