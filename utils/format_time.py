def get_format_time(seconds):
    min = int(seconds // 60)
    sec = int(seconds % 60)

    return f"{min:02d}:{sec:02d}"