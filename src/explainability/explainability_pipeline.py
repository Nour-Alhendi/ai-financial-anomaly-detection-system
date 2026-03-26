# explainability_pipeline.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from explainability.explainability_builder import run as run_builder
from explainability.llm_narrator       import run as run_llm
from explainability.finbert            import enrich_with_news
from explainability.report             import run as run_report
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def run():
    print("––– Layer 7: Explainability Pipeline –––")

    # 7A+B+C: SHAP + Narrative Engine + LLM placeholder
    print("[1/5] Explainability Builder (SHAP + Narrative Engine + LLM placeholder)")
    run_builder()

    # 7C: LLM Narrator summaries
    print("[2/5] LLM Narrator summaries")
    llm_path = ROOT / "data/explanations/explanations.parquet"
    run_llm(explanations_path=str(llm_path))

    # 7E: Enrich with FinBERT news sentiment
    print("[3/5] FinBERT news enrichment")
    df_llm = pd.read_parquet(ROOT / "data/explanations/llm_summaries.parquet")
    enriched_df = enrich_with_news(df_llm)
    enriched_df.to_parquet(ROOT / "data/explanations/llm_news_enriched.parquet", index=False)

    # 7D: HTML Report
    print("[4/5] Generate HTML report")
    run_report()

    print("––– Explainability pipeline complete –––")


if __name__ == "__main__":
    run()