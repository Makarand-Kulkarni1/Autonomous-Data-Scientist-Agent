from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import shutil
import json
from pathlib import Path

from src.orchestrator import run_pipeline
from src.report import generate_report, generate_pdf_report
from src.config import RUNS_DIR

app = FastAPI(title="Auto Data Scientist API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RUN_STATUS = {}


def _status_file(workspace: Path) -> Path:
    return workspace / "status.json"


def _write_status(workspace: Path, status: dict):
    with open(_status_file(workspace), "w", encoding="utf-8") as f:
        json.dump(status, f)


def _read_status(workspace: Path) -> dict | None:
    path = _status_file(workspace)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _workspace_for(run_id: str) -> Path:
    return RUNS_DIR / f"api_{run_id}" / "workspace"


def _set_status(run_id: str, workspace: Path, status: dict):
    RUN_STATUS[run_id] = status
    _write_status(workspace, status)


def _execute_run(run_id: str, csv_path: Path, goal: str, workspace: Path):
    try:
        result = run_pipeline(csv_path=csv_path, goal=goal, workspace=workspace, verbose=False)

        if result.success:
            report_path = generate_report(result, goal, workspace, workspace / "report.md")
            pdf_path = generate_pdf_report(result, goal, workspace, workspace / "report.pdf")
            _set_status(run_id, workspace, {
                "status": "success",
                "report_path": str(report_path),
                "pdf_report_path": str(pdf_path),
                "error": None,
            })
        else:
            last_log = result.logs[-1] if result.logs else None
            if last_log is None:
                detail = "No steps were attempted."
            elif not last_log.coder_success:
                detail = f"Code execution failed: {last_log.stderr.strip()[-500:]}"
            else:
                detail = f"Rejected by reviewer: {last_log.review_reason}"

            _set_status(run_id, workspace, {
                "status": "failed",
                "report_path": None,
                "pdf_report_path": None,
                "error": f"Pipeline failed at step {result.failed_at_step} ({last_log.title if last_log else '?'}). {detail}",
            })

    except Exception as e:
        _set_status(run_id, workspace, {
            "status": "failed",
            "report_path": None,
            "pdf_report_path": None,
            "error": str(e),
        })


@app.post("/runs")
async def start_run(background_tasks: BackgroundTasks,
                    goal: str = Form(...),
                    file: UploadFile = File(...)):
    run_id = uuid.uuid4().hex[:12]
    workspace = _workspace_for(run_id)
    workspace.mkdir(parents=True, exist_ok=True)

    csv_path = workspace.parent / "source_data.csv"
    with open(csv_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    _set_status(run_id, workspace, {"status": "running", "report_path": None, "pdf_report_path": None, "error": None})

    background_tasks.add_task(_execute_run, run_id, csv_path, goal, workspace)

    return {"run_id": run_id, "status": "running"}


@app.get("/runs/{run_id}/status")
async def get_status(run_id: str):
    if run_id in RUN_STATUS:
        return RUN_STATUS[run_id]

    workspace = _workspace_for(run_id)
    status = _read_status(workspace)
    if status is None:
        raise HTTPException(status_code=404, detail="Unknown run_id")

    RUN_STATUS[run_id] = status
    return status


@app.get("/runs/{run_id}/report")
async def get_report(run_id: str):
    workspace = _workspace_for(run_id)
    status = RUN_STATUS.get(run_id) or _read_status(workspace)

    if status is None:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    if status["status"] != "success":
        raise HTTPException(status_code=400, detail=f"Run is not finished successfully (status: {status['status']})")

    return FileResponse(status["report_path"], media_type="text/markdown", filename="report.md")


@app.get("/runs/{run_id}/report.pdf")
async def get_pdf_report(run_id: str):
    workspace = _workspace_for(run_id)
    status = RUN_STATUS.get(run_id) or _read_status(workspace)

    if status is None:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    if status["status"] != "success":
        raise HTTPException(status_code=400, detail=f"Run is not finished successfully (status: {status['status']})")

    return FileResponse(status["pdf_report_path"], media_type="application/pdf", filename="report.pdf")


@app.get("/health")
async def health():
    return {"status": "ok"}
