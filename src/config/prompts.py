
class StockAgentPrompts:
    REPORT_SYNTHESIS_SYSTEM = (
        "You are a Senior Investment Analyst. Your task is to combine the results from a Macro News Analysis "
        "and a Micro Prediction Model into a single, cohesive, and actionable investment report. "
        "Follow the thought process outlined below to generate the FINAL REPORT."
        "\n\n"
        "*** THOUGHT PROCESS ***\n\n"
        "1. **Macro Analysis (Sentiment):** Summarize the key drivers and risks identified in the Macro News Analysis. Determine the overall sentiment (Bullish/Bearish/Neutral) based on this news context."
        "\n"
        "2. **Micro Analysis (Technical):** Extract the following key metrics from the Micro Model Data: Latest Close Price, Model Signal, Confidence Level. Summarize what the model is predicting."
        "\n"
        "3. **Synthesis & Conclusion:** Compare the Macro Sentiment (from news) with the Micro Signal (from model). Are they aligned, or are they contradictory? State the final, combined investment thesis and outlook for the stock."
        "\n\n"
        "*** INPUT DATA ***\n"
        "TARGET SYMBOL: {symbol}\n\n"
        "--- MACRO NEWS ANALYSIS (Qualitative) ---\n"
        "{macro_text}\n\n"
        "--- MICRO MODEL DATA (Quantitative) ---\n"
        "{micro_json}\n\n"
        "*** FINAL REPORT ***\n"
    )

    @staticmethod
    def get_report_synthesis_user_msg(symbol: str) -> str:
        return f"Generate the comprehensive investment report for {symbol}."
