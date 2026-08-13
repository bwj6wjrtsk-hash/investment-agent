from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from threading import Lock
from uuid import uuid4

from .orchestrator import ResearchOrchestrator
from .schemas import ResearchRequest

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="company-research")
_JOBS: dict[str, dict] = {}
_LOCK = Lock()
_MAX_JOBS = 20


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _update(task_id: str, **values) -> None:
    with _LOCK:
        if task_id in _JOBS:
            _JOBS[task_id].update(values)
            _JOBS[task_id]["updated_at"] = _now()


def _run(task_id: str, request: ResearchRequest) -> None:
    try:
        _update(task_id, status="running", stage="research", progress=2,
                message="正在收集公司、财务与市场资料")
        orchestrator = ResearchOrchestrator()
        state = orchestrator.run(
            request,
            on_progress=lambda stage, progress, message: _update(
                task_id, stage=stage, progress=progress, message=message
            ),
        )
        _update(
            task_id, status="succeeded", stage="completed", progress=100,
            message="研究完成", analysis_id=state.analysis_id,
            result=state.model_dump(mode="json"),
        )
    except Exception as exc:
        _update(task_id, status="failed", message="研究失败", error=str(exc))


def start_job(payload: dict) -> dict:
    request = ResearchRequest.model_validate(payload)
    task_id = uuid4().hex
    job = {
        "task_id": task_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "message": "等待研究任务执行",
        "company": request.company,
        "symbol": request.symbol,
        "search_depth": request.search_depth,
        "analysis_id": "",
        "result": None,
        "error": "",
        "created_at": _now(),
        "updated_at": _now(),
    }
    with _LOCK:
        completed = [
            key for key, value in _JOBS.items()
            if value["status"] in {"succeeded", "failed"}
        ]
        while len(_JOBS) >= _MAX_JOBS and completed:
            _JOBS.pop(completed.pop(0), None)
        if len(_JOBS) >= _MAX_JOBS:
            raise RuntimeError("研究任务队列已满，请稍后再试")
        _JOBS[task_id] = job
    _EXECUTOR.submit(_run, task_id, request)
    return deepcopy(job)


def get_job(task_id: str) -> dict | None:
    with _LOCK:
        job = _JOBS.get(task_id)
        return deepcopy(job) if job else None
