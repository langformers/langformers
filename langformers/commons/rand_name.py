def get_name(identifier):
    """Generates names based on timestamp given an identifier."""
    from datetime import datetime

    name = f"langformers-{identifier}"
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")

    rand_name = f"{name}-d{date_str}-t{time_str}"

    return rand_name
