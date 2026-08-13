from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from uuid import uuid4

from .agents import AGENT_SEQUENCE, ResearchAgent
from .llm import DeepSeekClient
from .markdown_report import export_research_markdown
from .presentation import build_natural_language_presentation
from .repository import ResearchRepository
from .schemas import ResearchRequest, ResearchState
from .sources import SourceCollector


ProgressCallback = Callable[[str, int, str], None]
_LOGGER = logging.getLogger(__name__)
_STAGE_LABELS = {
    "research": "正在收集并整理事实证据",
    "financial": "正在核对历史财务事实",
    "industry": "正在分析行业与产业链变化",
    "company": "正在映射行业机会与公司能力",
    "story": "正在构建公司故事树",
    "expectation": "正在由当前价格反推隐含盈利与增长路径",
    "valuation": "正在运行程序化三年财务桥与自适应估值引擎",
    "catalyst": "正在识别可触发预期重定价的催化剂",
    "bear": "正在攻击故事因果链并执行反方检验",
    "mispricing": "正在识别市场预期差、验证链和潜在投资机会",
    "judge": "正在结合情景概率和预期差结论进行最终判断",
}


class ResearchOrchestrator:
    """固定 DAG：Research → Financial → Industry → Company → Story → Expectation → Valuation → Catalyst → Bear → Mispricing → Judge。"""

    def __init__(self, repository: ResearchRepository | None = None) -> None:
        self.repository = repository or ResearchRepository()

    @staticmethod
    def _notify(callback: ProgressCallback | None, stage: str, progress: int) -> None:
        if callback:
            try:
                callback(stage, progress, _STAGE_LABELS.get(stage, stage))
            except Exception:
                pass

    def run(
        self,
        request: ResearchRequest,
        analysis_id: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> ResearchState:
        analysis_id = analysis_id or uuid4().hex
        state = ResearchState(analysis_id=analysis_id, request=request)
        self.repository.create_analysis(analysis_id, request)
        try:
            self._notify(on_progress, "research", 3)
            llm = DeepSeekClient()
            research = ResearchAgent(llm, SourceCollector()).run(request)
            state.research = research
            self.repository.save_stage(analysis_id, "research", research)

            total = len(AGENT_SEQUENCE) + 1
            for index, (stage, agent_type, input_names) in enumerate(AGENT_SEQUENCE, start=1):
                self._notify(on_progress, stage, int(index / total * 90))
                payload = {name: getattr(state, name).model_dump(mode="json") for name in input_names}
                output = agent_type(llm).run(payload)
                setattr(state, stage, output)
                self.repository.save_stage(analysis_id, stage, output)

            # Stage saves remain independently durable; this final idempotent write
            # combines expectation inputs with valuation-derived Agent Forecast rows.
            if state.expectation is not None and state.valuation is not None:
                self.repository.save_expectation_series(
                    analysis_id, state.expectation, state.valuation
                )

            # 独立的投资者可读性层只翻译和解释现有结构化结果，
            # 不参与数据、预测、概率、估值或任何检查关卡计算。
            state.presentation = build_natural_language_presentation(state)
            self.repository.save_stage(analysis_id, "presentation", state.presentation)

            state.status = "completed"
            state.completed_at = datetime.now()
            self.repository.complete_analysis(analysis_id)
            try:
                report_path = export_research_markdown(state)
                _LOGGER.info("公司研究Markdown报告已生成：%s", report_path)
            except Exception:
                _LOGGER.exception("公司研究已完成，但Markdown报告生成失败：%s", analysis_id)
            self._notify(on_progress, "completed", 100)
            return state
        except Exception as exc:
            state.status = "failed"
            state.error = str(exc)
            self.repository.fail_analysis(analysis_id, str(exc))
            raise
