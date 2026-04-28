"""
FlashVLA — High-performance VLA inference engine.

Public exports (stable API — see ``docs/stable_api.md``):

    flash_vla.load_model(...)   → VLAModel
    flash_vla.VLAModel          — unified inference wrapper

Supported models: Pi0.5, Pi0, Pi0-FAST, GROOT N1.6.
Supported hardware: Jetson Thor (SM110), RTX 5090 (SM120), RTX 4090 (SM89).

Extending with new models: see ``docs/plugin_model_template.md``.

Usage::

    import flash_vla

    model = flash_vla.load_model(
        checkpoint="/path/to/checkpoint",
        framework="torch",
        autotune=3,
    )

    actions = model.predict(images=[base_img, wrist_img],
                            prompt="pick up the red block")
"""

__version__ = "0.1.0"

from flash_vla.api import load_model, VLAModel

__all__ = ["load_model", "VLAModel"]
