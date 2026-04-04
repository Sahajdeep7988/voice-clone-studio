"""
Unified FastAPI entrypoint.
Extends api/main.py with additional routers without modifying existing routes.

Run:
  uvicorn api.app:app --reload
"""

from fastapi.middleware.cors import CORSMiddleware

from api.main import app                               # all existing routes intact
from api.two_step_router import router                 # prepare / confirm (two-step pipeline)
from api.auth_router import router as auth_router      # auth: login, register, refresh, reset
from api.segments_router import router as seg_router   # segment curation
from api.upload_router import router as upload_router  # file upload

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
      "http://localhost:1420",
      "http://127.0.0.1:1420",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(auth_router)
app.include_router(seg_router)
app.include_router(upload_router)
