"""Persona loading and rendering (brought over from .archive/tests/test_units.py, English only)."""
from __future__ import annotations

from eva.personas import TEMPLATE_SLOTS, list_personas, load_persona, render


def test_eva_persona_renders_every_slot() -> None:
    assert list_personas()[0] == "eva"
    eva = load_persona("eva")
    assert eva.lang == "en" and eva.slots_present() == set(TEMPLATE_SLOTS)
    prompt = render(eva, supports_audio_tags=False, memory_text="- Has a cat named Miso.", user_name="Sam",
                    tool_notes="", locked_language="English")
    assert not any("{" + slot + "}" in prompt for slot in TEMPLATE_SLOTS)
    assert "Sam" in prompt and "Miso" in prompt and "always answer in English" in prompt
    assert "No tools are connected right now." in prompt and "Never write bracketed stage directions" in prompt


def test_unknown_user_is_never_named() -> None:
    prompt = render(load_persona("eva"), supports_audio_tags=False, memory_text="", tool_notes="")
    assert "your friend" in prompt and "never guess or invent one" in prompt
