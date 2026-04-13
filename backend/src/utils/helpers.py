def status_to_message(status, fps=None):
    if fps is None:
        return f"FireVisionNet detected: {status}"
    return f"FireVisionNet detected: {status} | FPS={fps:.1f}"