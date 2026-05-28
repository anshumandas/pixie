"""Pixie — a local-first, auto-configuring image editing app.

Click an object to select it, then remove the background, erase the object,
restyle it (ink / shadow), or adjust brightness & contrast.

The app detects your hardware at runtime and auto-selects the best models it
can run, falling back to built-in OpenCV methods so it works on any machine
with no model downloads — and upgrades itself when a GPU, model weights, or a
cloud backend become available.
"""

__version__ = "0.1.0"
