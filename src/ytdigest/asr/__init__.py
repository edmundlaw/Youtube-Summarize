"""ASR engines. Optional: `uv pip install -e '.[asr]'`.

Kept a package rather than a module because `runner._load_asr` already imports
`ytdigest.asr.qwen3` by name, and a second engine would sit beside it.
"""
