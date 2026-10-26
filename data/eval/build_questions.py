"""Writes questions.jsonl, the evaluation set: `uv run python data/eval/build_questions.py`.

Every reference answer and page was checked against the parsed reports. Pages are physical PDF
pages (1-based), the numbering citations use. Infosys figures come from its Form 20-F (US$).
"""

import json
from pathlib import Path


def q(id, question, reference, pages, kind="numeric", vectorless=False):
    pages = [{"report_id": r, "page": p} for r, p in pages]
    rids = sorted({p["report_id"] for p in pages})
    return {
        "id": id,
        "question": question,
        "reference": reference,
        "reference_pages": pages,
        "companies": sorted({r.split("_")[0] for r in rids}),
        "fiscal_years": sorted({r.split("_")[1] for r in rids}),
        "kind": kind,
        "vectorless": vectorless,
    }


I24, I25, T24 = "indigo_FY24", "indigo_FY25", "tcs_FY24"
F24, F25 = "infosys_FY24", "infosys_FY25"

QUESTIONS = [
    # --- InterGlobe Aviation (IndiGo) FY24: also the vectorless subset ---
    q(
        "indigo24-revenue",
        "What was InterGlobe Aviation's consolidated revenue from operations in FY24 and how "
        "much did it grow?",
        "Consolidated revenue from operations was Rs. 689,043.42 million in FY 2024, up from "
        "Rs. 544,464.53 million in FY 2023, a growth of about 26.6%.",
        [(I24, 59)],
        "table",
        True,
    ),
    q(
        "indigo24-profit",
        "What net profit did IndiGo report for FY24 compared with FY23?",
        "IndiGo reported a consolidated net profit of Rs. 81,724.68 million in FY 2024 against a "
        "net loss of Rs. 3,057.89 million in FY 2023.",
        [(I24, 59)],
        "numeric",
        True,
    ),
    q(
        "indigo24-passengers",
        "How many passengers did IndiGo carry in FY24?",
        "IndiGo carried 106.42 million passengers in FY 2024, its first year above 100 million, "
        "a 24% increase over 85.59 million in FY 2023.",
        [(I24, 60)],
        "numeric",
        True,
    ),
    q(
        "indigo24-load-factor",
        "What was IndiGo's passenger load factor in FY24?",
        "The passenger load factor rose from 82.1% in FY 2023 to 85.9% in FY 2024.",
        [(I24, 60)],
        "numeric",
        True,
    ),
    q(
        "indigo24-fleet-a321",
        "How many Airbus A321neo aircraft were in IndiGo's fleet as of March 31, 2024?",
        "As of March 31, 2024 the fleet included 94 Airbus A321neo aircraft (alongside 192 "
        "A320neo, 20 A320ceo, 45 ATR, 3 A321 freighters and damp-leased B777 and A320 aircraft).",
        [(I24, 60)],
        "fact",
        True,
    ),
    q(
        "indigo24-dividend",
        "Did IndiGo's board recommend a dividend for FY24?",
        "No. Based on performance, cash flow and losses carried forward from previous years, the "
        "Board did not recommend any dividend on equity shares for FY 2024.",
        [(I24, 61)],
        "fact",
        True,
    ),
    q(
        "indigo24-cask",
        "What was IndiGo's cost per available seat kilometre (CASK) in FY24 and why did it change?",
        "CASK decreased by 9.3% from Rs. 4.83 in FY 2023 to Rs. 4.38 in FY 2024, primarily driven "
        "by lower fuel cost and lower foreign exchange losses.",
        [(I24, 49)],
        "numeric",
        True,
    ),
    q(
        "indigo24-rask",
        "How did IndiGo's RASK change in FY24?",
        "RASK increased by 3.2% from Rs. 4.80 in FY 2023 to Rs. 4.96 in FY 2024, driven by higher "
        "yields and passenger load.",
        [(I24, 49)],
        "numeric",
        True,
    ),
    q(
        "indigo24-fuel",
        "How much did IndiGo spend on aircraft fuel in FY24 and how did it compare with capacity "
        "growth?",
        "Aircraft fuel expenses rose only 1.1%, from Rs. 236,460.17 million to Rs. 239,045.78 "
        "million, against a 21.8% increase in capacity, mainly because ATF prices fell.",
        [(I24, 49)],
        "numeric",
        True,
    ),
    q(
        "indigo24-cash",
        "What was IndiGo's total cash position at the end of FY24?",
        "Total cash rose 48.3% to Rs. 347,375.29 million as of March 31, 2024: free cash of "
        "Rs. 208,229.91 million and restricted cash of Rs. 139,145.38 million.",
        [(I24, 49)],
        "numeric",
        True,
    ),
    q(
        "indigo24-auditor",
        "Who are the statutory auditors of InterGlobe Aviation?",
        "M/s S.R. Batliboi & Co. LLP, Chartered Accountants, appointed at the 16th AGM on "
        "August 27, 2019.",
        [(I24, 64)],
        "fact",
        True,
    ),
    q(
        "indigo24-employees",
        "How many permanent employees did IndiGo have at the end of FY24?",
        "The Company had 36,860 permanent employees on its rolls as on March 31, 2024.",
        [(I24, 76)],
        "numeric",
        True,
    ),
    q(
        "indigo24-scope1",
        "What were IndiGo's Scope 1 greenhouse gas emissions in FY24?",
        "Total Scope 1 emissions were 8,414,458.31 tonnes of CO2 equivalent in FY 2024, up from "
        "6,789,520.00 in FY 2023.",
        [(I24, 128)],
        "table",
        True,
    ),
    q(
        "indigo24-csr",
        "How much did IndiGo spend on CSR in FY24?",
        "IndiGo spent Rs. 19,811,186 on CSR in FY 2024 although its statutory obligation (two "
        "percent of average net profit) was nil, so the whole amount was in excess of the "
        "requirement.",
        [(I24, 71)],
        "table",
        True,
    ),
    q(
        "indigo24-ops",
        "What were IndiGo's on-time performance and peak daily flights in FY24?",
        "On-time performance was 81.3% on DGCA metros, and peak operations reached 2,016 daily "
        "flights, 11% more than the 1,815 peak in FY 2023.",
        [(I24, 60)],
        "numeric",
        True,
    ),
    q(
        "indigo24-orders",
        "How many aircraft did IndiGo have on order according to its FY24 annual report?",
        "IndiGo had 985 aircraft on order, securing deliveries until the middle of the next "
        "decade, with an aim to become a 600+ aircraft carrier by 2030.",
        [(I24, 4), (I24, 5)],
        "fact",
        True,
    ),
    q(
        "indigo24-aog",
        "What did IndiGo account for in FY24 regarding aircraft grounded due to engine "
        "unavailability?",
        "IndiGo recorded an accrual for compensation from the original equipment manufacturer "
        "(OEM) for the ongoing Aircraft on Ground (AOG) situation caused by unavailability of "
        "engines, under Ind AS 115 and the existing contract, while a customised compensation "
        "plan was still being negotiated.",
        [(I24, 219)],
        "qualitative",
        True,
    ),
    # --- IndiGo FY25 ---
    q(
        "indigo25-revenue",
        "What was IndiGo's revenue from operations in FY25?",
        "Consolidated revenue from operations was INR 808,029 million in FY 2025 versus "
        "INR 689,043 million in FY 2024, growth of about 17.3%.",
        [(I25, 57)],
        "table",
    ),
    q(
        "indigo25-profit",
        "How did IndiGo's net profit change in FY25?",
        "Consolidated net profit decreased 11% to INR 72,584 million in FY 2025 from "
        "INR 81,725 million in FY 2024.",
        [(I25, 57)],
    ),
    q(
        "indigo25-passengers",
        "How many passengers did IndiGo carry in FY25?",
        "IndiGo carried 119.77 million passengers in FY 2025, 12.5% more than 106.42 million in "
        "FY 2024; since November 2024 it has carried over 10 million passengers every month.",
        [(I25, 58)],
    ),
    q(
        "indigo25-dividend",
        "What dividend did IndiGo's board recommend for FY25?",
        "A final dividend of INR 10 (100%) per fully paid equity share of INR 10 each, subject to "
        "shareholder approval at the AGM.",
        [(I25, 59)],
        "fact",
    ),
    q(
        "indigo25-cask",
        "What happened to IndiGo's CASK in FY25?",
        "CASK increased 6.4% from Rs. 4.38 in FY 2024 to Rs. 4.66 in FY 2025, mainly due to "
        "escalation in maintenance charges, higher airport charges and higher depreciation, "
        "partly offset by lower fuel prices.",
        [(I25, 49)],
    ),
    q(
        "indigo25-employees",
        "How many permanent employees did IndiGo have at the end of FY25?",
        "41,049 permanent employees as on March 31, 2025.",
        [(I25, 73)],
    ),
    q(
        "indigo25-cash",
        "What was IndiGo's total cash at the end of FY25?",
        "Total cash increased 38.7% to Rs. 481,705 million as of March 31, 2025, with free cash "
        "of Rs. 331,531 million and restricted cash of Rs. 150,174 million.",
        [(I25, 49)],
    ),
    # --- TCS FY24 ---
    q(
        "tcs24-revenue",
        "What was TCS's revenue in FY24 and its growth rate?",
        "TCS revenue from operations was Rs. 240,893 crore in FY 2024, up 6.8% from Rs. 225,458 "
        "crore in FY 2023 (versus 17.6% growth the year before).",
        [(T24, 75)],
        "table",
    ),
    q(
        "tcs24-margin",
        "What was TCS's operating (EBIT) margin in FY24?",
        "EBIT margin was 24.6% in FY 2024 excluding a one-time item (24.2% as reported), "
        "compared with 24.1% in FY 2023, helped by productivity, utilisation and lower "
        "subcontractor costs.",
        [(T24, 75)],
        "table",
    ),
    q(
        "tcs24-profit",
        "What was TCS's net profit attributable to shareholders in FY24?",
        "Profit after tax attributable to shareholders was Rs. 45,908 crore as reported "
        "(Rs. 46,585 crore excluding a one-time item), up from Rs. 42,147 crore in FY 2023.",
        [(T24, 75)],
        "table",
    ),
    q(
        "tcs24-dividend",
        "What dividend and buyback did TCS return to shareholders for FY24?",
        "A final dividend of Rs. 28 per share took the total dividend for the year to Rs. 73 per "
        "share, and TCS completed its fifth buyback returning Rs. 17,000 crore.",
        [(T24, 8), (T24, 11)],
        "fact",
    ),
    q(
        "tcs24-employees",
        "How many employees did TCS have in FY24 and what was its attrition?",
        "TCS had 601,546 employees, with LTM attrition in IT services of 12.5%, down 760 bps "
        "over the previous year.",
        [(T24, 9)],
    ),
    q(
        "tcs24-tcv",
        "What was TCS's total contract value (TCV) of deal wins in FY24?",
        "TCV of deal wins was US$ 42.7 billion, double-digit growth.",
        [(T24, 8)],
    ),
    q(
        "tcs24-diversity",
        "What share of TCS's workforce were women in FY24?",
        "Women made up 35.6% of the employee base, and the workforce spanned over 152 "
        "nationalities.",
        [(T24, 9)],
        "fact",
    ),
    q(
        "tcs24-growth-why",
        "Why was TCS's revenue growth muted in FY24 despite record deal wins?",
        "New deals converted into revenue as expected, but those inflows were offset by "
        "reductions in the existing revenue base as earlier projects were completed, optimised "
        "or downsized, amid macro uncertainty and geopolitical volatility.",
        [(T24, 75)],
        "qualitative",
    ),
    # --- Infosys (Form 20-F, US$) ---
    q(
        "infosys24-revenue",
        "What were Infosys's revenues in fiscal 2024 and how much did they grow?",
        "Revenues were $18,562 million in fiscal 2024, up $350 million or 1.9% from $18,212 "
        "million in fiscal 2023.",
        [(F24, 62)],
        "table",
    ),
    q(
        "infosys24-profit",
        "What was Infosys's net profit in fiscal 2024?",
        "Net profit was $3,169 million in fiscal 2024.",
        [(F24, 61), (F24, 36)],
    ),
    q(
        "infosys24-employees",
        "How many employees did Infosys have as of March 31, 2024?",
        "Infosys had 317,240 employees as of March 31, 2024, up from 242,371 in 2020.",
        [(F24, 36)],
    ),
    q(
        "infosys24-fixed-price",
        "What share of Infosys's fiscal 2024 revenue came from fixed-price contracts?",
        "About 53% in fiscal 2024 (52% in fiscal 2023); products and platforms were 5.5% of "
        "revenue.",
        [(F24, 64)],
    ),
    q(
        "infosys25-revenue",
        "What were Infosys's revenues in fiscal 2025 and the growth in reported and constant "
        "currency?",
        "Revenues were $19,277 million in fiscal 2025, growth of 3.9% in reported currency (from "
        "$18,562 million) and 4.2% in constant currency.",
        [(F25, 65), (F25, 66)],
    ),
    q(
        "infosys25-employees",
        "How many employees did Infosys have as of March 31, 2025?",
        "323,578 employees as of March 31, 2025.",
        [(F25, 38)],
    ),
    q(
        "infosys25-leaders",
        "Who were Infosys's chairman and CEO according to its fiscal 2025 annual report?",
        "Nandan Nilekani was the non-executive, non-independent Chairman and Salil Parekh the "
        "Chief Executive Officer and Managing Director.",
        [(F25, 81)],
        "fact",
    ),
    q(
        "infosys25-usd-share",
        "What share of Infosys's fiscal 2025 revenues was denominated in US dollars?",
        "64.2% of revenues in fiscal 2025 were in US dollars, down from 66.2% in fiscal 2024 "
        "(the euro share rose to 16.8% from 14.8%).",
        [(F25, 71)],
        "table",
    ),
    # --- Cross-company / cross-year comparisons ---
    q(
        "cmp-indigo-infosys-growth",
        "Compare the revenue growth of InterGlobe Aviation and Infosys in FY24.",
        "IndiGo's revenue from operations grew about 26.6% (Rs. 544,465 million to Rs. 689,043 "
        "million) while Infosys's revenues grew 1.9% ($18,212 million to $18,562 million), so "
        "IndiGo grew far faster.",
        [(I24, 59), (F24, 62)],
        "comparison",
    ),
    q(
        "cmp-indigo-tcs-growth",
        "Compare the revenue growth of IndiGo and TCS in FY24.",
        "IndiGo grew revenue about 26.6% to Rs. 689,043 million, while TCS grew 6.8% to "
        "Rs. 240,893 crore.",
        [(I24, 59), (T24, 75)],
        "comparison",
    ),
    q(
        "cmp-indigo-yoy",
        "How did IndiGo's revenue growth in FY25 compare with FY24?",
        "Growth slowed from about 26.6% in FY 2024 (to Rs. 689,043 million) to about 17.3% in "
        "FY 2025 (to Rs. 808,029 million).",
        [(I24, 59), (I25, 57)],
        "comparison",
    ),
    q(
        "cmp-indigo-profit-vs-revenue",
        "In FY25, did IndiGo's profit move in the same direction as its revenue?",
        "No. Revenue grew about 17% to INR 808,029 million but net profit fell 11% to "
        "INR 72,584 million from INR 81,725 million.",
        [(I25, 57)],
        "comparison",
    ),
    q(
        "cmp-infosys-yoy",
        "How did Infosys's revenue growth in fiscal 2025 compare with fiscal 2024?",
        "Growth improved from 1.9% in fiscal 2024 ($18,562 million) to 3.9% in fiscal 2025 "
        "($19,277 million).",
        [(F24, 62), (F25, 65)],
        "comparison",
    ),
    q(
        "cmp-headcount",
        "Compare the employee headcount of TCS, Infosys and IndiGo at the end of FY24.",
        "TCS had 601,546 employees, Infosys 317,240 and IndiGo 36,860 permanent employees, so "
        "TCS was almost twice Infosys and about 16 times IndiGo.",
        [(T24, 9), (F24, 36), (I24, 76)],
        "comparison",
    ),
    q(
        "cmp-dividend",
        "Compare the FY24 dividends of IndiGo and TCS.",
        "IndiGo did not recommend any dividend for FY 2024 because of losses carried forward, "
        "while TCS paid a total dividend of Rs. 73 per share (including a Rs. 28 final dividend) "
        "plus a Rs. 17,000 crore buyback.",
        [(I24, 61), (T24, 8)],
        "comparison",
    ),
    q(
        "cmp-infosys-headcount",
        "How did Infosys's headcount change between March 2024 and March 2025?",
        "It rose from 317,240 to 323,578 employees, an increase of 6,338 (about 2%).",
        [(F24, 36), (F25, 38)],
        "comparison",
    ),
]

if __name__ == "__main__":
    out = Path(__file__).with_name("questions.jsonl")
    ids = [x["id"] for x in QUESTIONS]
    assert len(ids) == len(set(ids)), "duplicate ids"
    with open(out, "w", encoding="utf-8") as f:
        for item in QUESTIONS:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    n_vl = sum(x["vectorless"] for x in QUESTIONS)
    print(f"{len(QUESTIONS)} questions ({n_vl} vectorless) -> {out}")
