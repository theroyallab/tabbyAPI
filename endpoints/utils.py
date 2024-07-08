import os

do_export_openapi = os.getenv("EXPORT_OPENAPI", "").lower() in ("true", "1")
