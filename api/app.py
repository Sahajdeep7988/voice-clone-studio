"""
Unified FastAPI entrypoint.
Extends api/main.py with the two-step pipeline routes without modifying it.

Run:
  uvicorn api.app:app --reload
"""

from api.main import app                    # all existing routes intact
from api.two_step_router import router      # new two-step routes
from api.auth_router import router as auth_router  # auth routes

app.include_router(router)
app.include_router(auth_router)
