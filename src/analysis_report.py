import os
from glob import glob
from datetime import datetime
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import logger


def generate_final_report():
    logger.info("Compiling final markdown report from analysis outputs...")

    section_keywords = {
        "Executive Summary": ["executive_summary"],
        "Descriptive Statistics and EDA": [
            "association_stats", "correlation_heatmap", "violin",
            "pairplot", "ae_rates_bar", "onset_boxplot",
            "age_histogram", "sankey"
        ],
        "Causal Inference Analysis": [
            "psm", "tmle", "causal_forest", "doubleml"
        ],
        "Predictive Modeling and SHAP Explainability": [
            "model_evaluation", "logistic_regression_summary", "bayesian_logistic_summary",
            "shap_summary", "rfe_summary"
        ],
        "Advanced Analysis (Clustering, Survival, Anomalies)": [
            "kaplan_meier", "clustering", "anomaly"
        ]
    }

    sections = {section: [] for section in section_keywords}
    sections["Appendices"] = []

    report_file = "reports/final_analysis_report.md"
    markdown_files = sorted(glob("summaries/**/*.md", recursive=True))

    def categorize_file(fname):
        for section, keywords in section_keywords.items():
            if any(k in fname for k in keywords):
                return section
        return "Appendices"

    for md_file in markdown_files:
        section = categorize_file(md_file)
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            # Avoid reintroducing duplicate headers
            content = content.replace("# Final Vaccine AE Causal Analysis Report", "")
            content = content.replace("# Final TAK Vaccine AE Safety Analysis Report", "")
            content = content.replace("---", "")
            title = os.path.basename(md_file).replace('_', ' ').replace('.md', '').title()
            if not content.startswith("##"):
                sections[section].append(f"## {title}\n\n{content}")
            else:
                sections[section].append(content)

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Final TAK Vaccine AE Safety Analysis Report\n")
        f.write(f"\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n---\n")

        for section, contents in sections.items():
            if contents:
                f.write(f"\n# {section}\n")
                f.write("\n\n".join(contents))
                f.write("\n\n")

    logger.info(f"✅ Final report compiled to: {report_file}")
    
generate_final_report()