import os

import requests
import streamlit as st

API_URL = os.getenv("RESEARCH_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="公司预期变化检测", layout="wide")
st.title("上市公司预期变化检测系统")
st.caption("信息 → 事实 → 故事 → 市场预期 → 盈利 → 估值 → 证伪；不是自动荐股。")

with st.form("research"):
    company = st.text_input("公司名称或股票代码", placeholder="例如：中兴通讯 / 000063")
    symbol = st.text_input("股票代码（可选）", placeholder="000063")
    depth = st.selectbox("检索深度", ["basic", "deep"])
    submitted = st.form_submit_button("开始研究", type="primary")

if submitted:
    if not company.strip():
        st.error("请输入公司名称或股票代码")
    else:
        with st.spinner("正在按固定 DAG 运行 8 个 Agent 与 Judge..."):
            try:
                response = requests.post(
                    f"{API_URL}/analyses",
                    json={"company": company, "symbol": symbol, "search_depth": depth},
                    timeout=900,
                )
                response.raise_for_status()
                st.session_state["result"] = response.json()
            except Exception as exc:
                st.error(f"研究失败：{exc}")


result = st.session_state.get("result")
if result:
    judge = result.get("judge") or {}
    st.success(f"研究完成 · Analysis ID: {result.get('analysis_id', '')}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("故事成立概率", f"{judge.get('story_probability', 0):.0%}")
    c2.metric("基本面兑现概率", f"{judge.get('fundamental_delivery_probability', 0):.0%}")
    c3.metric("估值扩张概率", f"{judge.get('valuation_expansion_probability', 0):.0%}")
    c4.metric("正向预期差", "★" * int(judge.get("positive_expectation_gap_score", 1)))
    c5.metric("风险", "★" * int(judge.get("risk_score", 1)))

    labels = [
        ("一、公司现在是什么", "company"),
        ("二、过去发生了什么", "financial"),
        ("三、行业正在发生什么", "industry"),
        ("四、公司为什么可能受益", "company"),
        ("五、市场现在交易什么", "expectation"),
        ("六、哪些已经 Price In", "expectation"),
        ("七、潜在的新故事", "story"),
        ("八、新故事成立需要什么", "story"),
        ("九、市场隐含盈利预期", "expectation"),
        ("十、Agent 盈利推演", "valuation"),
        ("十一、Bear Case", "bear"),
        ("十二、估值区间", "valuation"),
        ("十三、关键验证指标", "judge"),
        ("十四、最终判断", "judge"),
    ]
    tabs = st.tabs([label for label, _ in labels])
    for tab, (_, key) in zip(tabs, labels):
        with tab:
            st.json(result.get(key) or {}, expanded=True)

    evidence = (result.get("research") or {}).get("evidence", [])
    with st.expander(f"证据层（{len(evidence)} 条，可追溯来源/日期/置信度）"):
        for item in evidence:
            st.markdown(
                f"**{item.get('fact', '')}**  · {item.get('source', '')}  "
                f"· 置信度 {item.get('confidence', 0):.0%}"
            )
            if item.get("source_url"):
                st.link_button("查看原始来源", item["source_url"])
            st.code(item.get("excerpt", "")[:2000], language=None)

    st.warning(judge.get("disclaimer", "仅供研究，不构成投资建议。"))
