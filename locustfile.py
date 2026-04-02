import json
import os
import random
import time
from uuid import uuid4

from locust import HttpUser, task, between
from locust.exception import StopUser


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


AUTH_EMAIL = _env("LOCUST_AUTH_EMAIL")
AUTH_PASSWORD = _env("LOCUST_AUTH_PASSWORD")
INPUT_FILES = (_env("LOCUST_INPUT_FILES") or "").split(",")
INPUT_FILES = [p.strip() for p in INPUT_FILES if p.strip()]

MODEL_PATH = _env("LOCUST_MODEL_PATH")  # used for /inference/convert if desired
INPUT_AUDIO_PATH = _env("LOCUST_INPUT_AUDIO_PATH")
OUTPUT_DIR = _env("LOCUST_OUTPUT_DIR", "/tmp/locust_outputs")

SHARED_MODEL = _env("LOCUST_SHARED_MODEL", "shared_model")
RANDOM_MODEL_PREFIX = _env("LOCUST_RANDOM_MODEL_PREFIX", "model")

SSE_TERMINAL_EVENTS = {"complete", "error", "paused"}


def _random_model_name() -> str:
    return f"{RANDOM_MODEL_PREFIX}_{uuid4().hex[:8]}"


def _pick_files() -> list[str]:
    if not INPUT_FILES:
        return []
    if len(INPUT_FILES) == 1:
        return INPUT_FILES
    # random 1–3 files
    k = random.randint(1, min(3, len(INPUT_FILES)))
    return random.sample(INPUT_FILES, k)


class VoiceCloneUser(HttpUser):
    wait_time = between(0.5, 2.5)

    def on_start(self):
        if not AUTH_EMAIL or not AUTH_PASSWORD:
            self.environment.events.request.fire(
                request_type="LOCUST",
                name="missing_auth_env",
                response_time=0,
                response_length=0,
                exception=RuntimeError(
                    "Set LOCUST_AUTH_EMAIL and LOCUST_AUTH_PASSWORD for login"
                ),
            )
            raise StopUser()

        if not INPUT_FILES:
            self.environment.events.request.fire(
                request_type="LOCUST",
                name="missing_input_files",
                response_time=0,
                response_length=0,
                exception=RuntimeError(
                    "Set LOCUST_INPUT_FILES to one or more safe paths on the server"
                ),
            )
            raise StopUser()

        payload = {"email": AUTH_EMAIL, "password": AUTH_PASSWORD}
        with self.client.post("/auth/login", json=payload, catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"login failed: {r.status_code} {r.text}")
                raise StopUser()
            data = r.json()
            token = data.get("token")
            if not token:
                r.failure("login missing token")
                raise StopUser()
            self.token = token
            self.headers = {"Authorization": f"Bearer {token}"}
            r.success()

        self.session_ids: list[str] = []

    def _store_session_id_from_sse(self, text: str) -> None:
        for line in text.splitlines():
            if line.startswith("data:"):
                try:
                    payload = json.loads(line.split(":", 1)[1].strip())
                    sid = payload.get("session_id")
                    if sid:
                        self.session_ids.append(sid)
                        return
                except Exception:
                    continue

    @task(6)
    def create_session(self):
        files = _pick_files()
        if not files:
            return
        model_name = SHARED_MODEL if random.random() < 0.5 else _random_model_name()

        payload = {"files": files, "model_name": model_name}
        with self.client.post(
            "/sessions/create",
            json=payload,
            headers=self.headers,
            stream=True,
            catch_response=True,
        ) as r:
            if r.status_code != 200:
                r.failure(f"create_session: {r.status_code} {r.text}")
                return
            # Read a limited number of lines to extract session_id quickly,
            # then close to avoid long-running training in load tests.
            lines = []
            try:
                for _ in range(5):
                    line = r.iter_lines().__next__().decode("utf-8")
                    lines.append(line)
            except Exception:
                pass
            self._store_session_id_from_sse("\n".join(lines))
            r.success()

    @task(3)
    def get_sessions(self):
        with self.client.get("/sessions", headers=self.headers, catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"get_sessions: {r.status_code} {r.text}")
            else:
                r.success()

    @task(1)
    def stop_session(self):
        session_id = random.choice(self.session_ids) if self.session_ids else f"ghost-{uuid4().hex[:6]}"
        with self.client.post(
            f"/sessions/{session_id}/stop",
            json={"force": False},
            headers=self.headers,
            catch_response=True,
        ) as r:
            # 200 if active, 404 otherwise (acceptable under load)
            if r.status_code not in (200, 404):
                r.failure(f"stop_session: {r.status_code} {r.text}")
            else:
                r.success()

    @task(1)
    def sse_stream_until_terminal(self):
        files = _pick_files()
        if not files:
            return
        model_name = SHARED_MODEL  # intentional collision
        payload = {"files": files, "model_name": model_name}
        start = time.time()

        with self.client.post(
            "/sessions/create",
            json=payload,
            headers=self.headers,
            stream=True,
            catch_response=True,
        ) as r:
            if r.status_code != 200:
                r.failure(f"sse_stream: {r.status_code} {r.text}")
                return

            terminal = False
            try:
                for raw in r.iter_lines():
                    line = raw.decode("utf-8")
                    if line.startswith("data:"):
                        try:
                            data = json.loads(line.split(":", 1)[1].strip())
                            sid = data.get("session_id")
                            if sid:
                                self.session_ids.append(sid)
                        except Exception:
                            pass
                    if line.startswith("event:"):
                        event = line.split(":", 1)[1].strip()
                        if event in SSE_TERMINAL_EVENTS:
                            terminal = True
                            break
                    if time.time() - start > 15:
                        break
            finally:
                if terminal:
                    r.success()
                else:
                    r.failure("sse_stream: no terminal event before timeout")
