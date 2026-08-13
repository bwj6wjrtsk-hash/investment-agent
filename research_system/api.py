from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from .orchestrator import ResearchOrchestrator
from .repository import ResearchRepository
from .schemas import ResearchRequest

load_dotenv()

app = FastAPI(
    title="Expectation Change Research API",
    version="0.2.0",
    description="上市公司预期变化检测系统；输出研究证据与概率，不提供自动交易。",
)
repository = ResearchRepository()
orchestrator = ResearchOrchestrator(repository)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "expectation-change-research"}


@app.post("/analyses")
def create_analysis(request: ResearchRequest) -> dict:
    try:
        return orchestrator.run(request).model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/analyses")
def list_analyses(limit: int = 20) -> list[dict]:
    return repository.list_analyses(min(max(limit, 1), 100))


@app.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str) -> dict:
    result = repository.get_analysis(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="analysis not found")
    return result


@app.get("/companies/{company}/expectation-timeline")
def expectation_timeline(company: str) -> list[dict]:
    return repository.get_expectation_timeline(company)
