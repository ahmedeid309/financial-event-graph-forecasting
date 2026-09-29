from __future__ import annotations


ARTICLE_MAP_PROMPT_TEMPLATE = """
You are Stage 1 of an evidence-first financial Event Knowledge Graph pipeline.
Return one valid JSON object only. Normal formatting with whitespace/indentation is fine -- do not
spend any effort minifying it. Do not include Markdown or prose outside JSON.
Start your response with {{ and end it with }}.

Goal:
- Map the article before event extraction.
- Identify companies, securities, and compact evidence spans that may support concrete financial or business events.
- Select exact evidence spans from the article text. Do not summarize them.
- Evidence spans may include reported metrics, market moves, guidance, risks, operations, product/business facts,
  competition, supply/customer relations, portfolio facts, or causal statements.
- Do not skip the lead paragraph or named body subsections in favor of only estimate, valuation, stock-return,
  or rating paragraphs. The article's main operating/business announcement usually appears before the market
  estimate section and must be mapped when it is company-specific.
- Distinguish concrete company-specific business facts from vague editorial thesis.
- If one company section contains several distinct numbers or metrics, include enough evidence spans so each
  metric family remains visible to the event extractor. Do not hide multiple facts inside a broad summary.
- Do not mark publisher promotion, author disclosure, legal disclaimer, related links, or footer text as financial evidence.
- Keep each span focused on one sentence or a short adjacent pair of sentences.

JSON schema:
{{
  "article_id": "{article_id}",
  "source_ticker": "{source_ticker}",
  "date": "{date}",
  "article_topic": "",
  "companies": [
    {{"name": "", "ticker": "", "role": ""}}
  ],
  "other_entities": [],
  "tracked_anchor_company_mentions": [],
  "event_areas_to_extract": [],
  "financial_evidence_spans": [
    {{
      "span_id": "S1",
      "text": "",
      "companies_or_assets": [],
      "tickers": [],
      "contains_financial_fact": true,
      "contains_temporal_info": false,
      "contains_causal_info": false,
      "numbers_or_money_mentions": [],
      "reason": ""
    }}
  ],
  "article_structure_notes": ""
}}

Article metadata:
article_id: {article_id}
source_ticker: {source_ticker}
article_date: {date}
title: {title}
detected_tickers: {detected_tickers_text}

Tracked anchor companies:
{anchors_json}

Article text:
{text}
""".strip()


SECTION_CLASSIFICATION_PROMPT_TEMPLATE = """
You are the article-section classifier for a financial Event Knowledge Graph pipeline.
Return one valid JSON object only. Normal formatting with whitespace/indentation is fine -- do not
spend any effort minifying it. Do not include Markdown or prose outside JSON.
Start your response with {{ and end it with }}.

Goal:
- Divide the article text into meaningful sections for extraction quality control.
- Identify which parts can support real financial/business events and which parts are boilerplate.
- Do not extract events in this stage.
- Use exact short quotes from the article for start_quote and end_quote so the system can align sections back to text.

Allowed section_type values:
core_article_body, financial_analysis, author_disclosure, publisher_promotion,
legal_disclaimer, related_links, metadata_or_footer, unknown

Rules:
- core_article_body and financial_analysis are normally allowed for event extraction.
- unknown is allowed only when the text appears article-like but cannot be confidently classified.
- author_disclosure, publisher_promotion, legal_disclaimer, related_links, and metadata_or_footer are not allowed.
- Publisher text such as recommendations, newsletter ads, affiliate compensation, stock-pick promotions,
  author ownership/positions, board disclosures, and legal disclaimers must be disallowed.

JSON schema:
{{
  "article_id": "{article_id}",
  "article_sections": [
    {{
      "section_id": "SEC1",
      "section_type": "core_article_body",
      "allowed_for_event_extraction": true,
      "start_quote": "",
      "end_quote": "",
      "section_summary": ""
    }}
  ]
}}

Article metadata:
article_id: {article_id}
source_ticker: {source_ticker}
article_date: {date}
title: {title}

Article text:
{text}
""".strip()


EVENT_BLOCK_PROMPT_TEMPLATE = """
You are extracting financial event candidates for an evidence-first Event Knowledge Graph.
Return one valid JSON object only. Normal formatting with whitespace/indentation is fine -- do not
spend any effort minifying it. Do not include Markdown or prose outside JSON.
Start your response with {{ and end it with }}.

This is block {block_index} of {block_count} from the same article.
The block contains exact article evidence spans with span IDs and section metadata.

Goal:
- Extract every concrete financial or business event/fact supported by this block.
- Extract events for all companies, assets, indexes, and securities in the block, not only the source ticker.
- Create separate candidates for distinct facts, metrics, companies, periods, baselines, ranks, weights, or percentages.
- Create separate candidates for separate predicates in the same evidence sentence. If one clause is a current
  fact and another clause is a future expectation/risk/target/guidance with its own period, split them.
- Temporal complexity must never reduce event coverage. If the fact is concrete and supported but the controlling
  time is ambiguous, still extract the candidate and leave temporal_expression blank.
- When an evidence sentence contains several concrete facts, enumerate the facts first; assign temporal_expression
  second. Do not omit facts because their temporal expression is blank, current-state, comparative, or complex.
- Do not create mirrored duplicate candidates by switching company perspectives.
- Preserve exact numbers, percentages, money amounts, dates, periods, ranks, and weights.
- Put raw temporal wording in temporal_expression only when that wording directly modifies the same fact,
  metric, or clause as the candidate. If a time phrase belongs to a neighboring fact in the same sentence,
  leave temporal_expression blank for this candidate.
- If temporal_expression belongs to a different clause, either create a separate candidate for that clause or
  omit the temporal_expression; do not attach it to the wrong predicate.
- Do preserve same-clause reporting or guidance period wording such as full-year, year-to-date, a named year,
  a quarter, a month, or a since-window when it controls the candidate's metric, target, forecast, or reported fact.
- Do not use comparison baselines as event_time expressions. Phrases such as from a high/low/peak/trough,
  from an average, versus a peer/index, or compared with a baseline are baselines unless they also state the
  event's controlling measurement period.
- Use evidence_span_ids and evidence_text to point to the exact support.
- Keep output compact. Do not include reasoning, explanations, copied article-map context, or long prose inside
  any field. event_description should be one concise sentence under 160 characters when possible. event_trigger
  should be a short phrase. evidence_text should be the shortest exact quote that supports the candidate, usually
  one sentence or less and under 240 characters.
- financial_metric_mentions must capture BOTH the metric name and its exact value together, one object per
  metric: {{"metric": "dividend_yield", "value": "6.72%", "unit": "%", "time_period": "Q3"}}. Never emit a metric
  name without its value, and never emit a bare number without naming the metric it measures. Preserve the value
  exactly as written, including %, $, currency words, and units. Leave unit or time_period blank when not stated.
- participants and mentioned_entities together must include every company, fund, person, product, or organization
  named in the event_description or evidence_text: participants for entities acting in or directly party to the
  event, mentioned_entities for others merely named. Never drop a named counterparty. impacted_anchor_companies
  may be empty unless the quote explicitly states an anchor-company impact.
- Populate explicit_causal_relations whenever the evidence states a cause-effect link with wording such as
  "due to", "because", "led to", "drove", "as a result of", "thanks to", or "resulting in". Write each as a short
  "cause -> effect" string, e.g. "biosimilar competition -> Humira sales decline". Leave it empty only when no
  causal wording is present.
- In dense evidence blocks, preserve coverage by making each candidate compact rather than writing fewer verbose
  candidates. Never spend tokens explaining why a candidate is valid.
- Do not create events from promotional calls to action, author disclosures, generic newsletter performance,
  legal disclaimers, related links, or vague editorial framing.
- Do not create events from stock-screener or guru-strategy output, even when it looks concrete and numeric.
  This includes Validea/guru model scores, "rates highest among N guru strategies", "passes all six criteria",
  "passes all tests", "receives a 95% rating under the X strategy", named investor-strategy scorecards, and
  large-cap/mid-cap growth-or-value style classifications. These describe a screening algorithm's opinion,
  not something that happened to the company. Skip them entirely.
- Do not create events from fund or ETF mechanics: expense ratios, assets under management, fund
  inflows/outflows, top-10 holdings lists, cost-basis calculations, beta/standard-deviation statistics, or
  hypothetical "$10,000 invested since inception" figures. Skip them entirely.
- Do not create events from biographical or firm-profile trivia, such as who founded an investment firm or
  where a company is headquartered.
- Do not create events from vague qualitative positioning unless the same candidate includes a concrete action,
  metric, target, contract, order, reported result, rating, rank, or commitment.
- Do not create portfolio_holding events from publisher/author disclosure or recommendation statements.

Event typing:
- Use revenue_growth/profit_growth for revenue or income growth facts.
- Use portfolio_holding only for an actual security/asset ownership, allocation, or ownership change by a
  portfolio, fund, investor, institution, insider, or author. Do not use portfolio_holding for customer orders,
  purchase commitments, product deliveries, supply agreements, vendor/customer relationships, exclusivity
  clauses, or service deals; use partnership, supply_chain_event, business_segment_performance, or other.
- Use market_cap_milestone for market-cap facts.
- Use stock_performance for stock/share return facts.
- Use price_target_change for analyst/consensus target-price facts.
- Use supply_chain_event for supplier/customer or availability constraints.
- Use competitive_position for rivalry/substitution/market-position facts.
- Use business_segment_performance for operations, platform, product, subscriber/customer, service, or scale facts.
  This includes segment revenue/income facts and gross/operating margin facts.
- Use dividend_announcement for any dividend initiation, raise, cut, suspension, per-share amount, payout, or
  dividend-yield fact. Do not fall back to other for dividend facts.
- Use valuation_multiple for P/E, forward/trailing earnings multiples, price-to-book, EV/EBITDA, or
  "trades at N times earnings" facts.
- Use balance_sheet for debt levels, debt maturities, cash and equivalents, leverage, or debt-to-EBITDA facts.
- Use cash_flow for free cash flow, operating cash flow, cash burn, or cash-runway facts.
- Use analyst_estimate for consensus/analyst EPS or revenue estimates, estimate revisions, and earnings ESP.
- Use capital_allocation for capital expenditure, investment programs, and cost-reduction/savings targets.
- Use price_level_range for 52-week highs/lows, last-trade prices, and stock price levels.
- Use options_activity for put/call contracts, strike prices, covered calls, option premiums, or expirations.
- Use macroeconomic_event for Federal Reserve, PMI, CPI, inflation, interest-rate, or jobs-report facts.
- Use guidance_update for reiterated, reaffirmed, raised, or lowered company guidance/targets/outlooks.
- Use market_commentary only for broad market/index commentary.
- Use other only when no specific type fits.

Importance scoring (importance_score is an integer 1-3, and must differentiate events):
- 3 = material, potentially market-moving: mergers/acquisitions, guidance initiation or change, dividend
  initiation/cut/suspension, major legal or regulatory action, earnings surprise, large financing.
- 2 = notable company-specific result: reported revenue/profit growth or decline, segment or subscriber
  performance, analyst rating/target change, buyback, notable operational milestone.
- 1 = routine or background fact: a standing metric mention (current yield, P/E, cash balance, debt level),
  minor commentary, or context. Do not default every event to 1 -- separate material events from routine facts.

Allowed event_type values: {allowed_types}
Allowed anchor impact_type values: {allowed_impact_types}
Allowed anchor impact_direction values: {allowed_impact_directions}

JSON schema:
{{
  "article_id": "{article_id}",
  "source_ticker": "{source_ticker}",
  "date": "{date}",
  "block_id": "{block_id}",
  "event_candidates": [
    {{
      "event_type": "",
      "event_trigger": "",
      "event_description": "",
      "main_company": "",
      "ticker": "",
      "participants": [],
      "mentioned_entities": [],
      "financial_metric_mentions": [
        {{"metric": "", "value": "", "unit": "", "time_period": ""}}
      ],
      "temporal_expression": "",
      "explicit_causal_relations": [],
      "related_event_mentions": [],
      "impacted_anchor_companies": [
        {{
          "ticker": "",
          "company": "",
          "impact_type": "",
          "impact_direction": "neutral",
          "evidence": "",
          "raw_mention": ""
        }}
      ],
      "evidence_span_ids": [],
      "evidence_text": "",
      "importance_score": 1,
      "confidence": 0.0
    }}
  ]
}}

Article metadata:
article_id: {article_id}
source_ticker: {source_ticker}
article_date: {date}
title: {title}

Article map:
{article_map_json}

Tracked anchor companies:
{anchors_json}

Evidence block:
{block_json}
""".strip()


EVIDENCE_ADMISSIBILITY_PROMPT_TEMPLATE = """
You are the evidence admissibility judge for a financial Event Knowledge Graph pipeline.
Return one valid JSON object only. Normal formatting with whitespace/indentation is fine -- do not
spend any effort minifying it. Do not include Markdown or prose outside JSON.
Start your response with {{ and end it with }}.

Goal:
- Decide whether each event candidate is supported by admissible article evidence.
- Do not rewrite, merge, split, or create candidates.
- Admissibility means the candidate is an article-supported financial, market, operating, valuation,
  guidance, competitive, legal, strategic, or business fact. It does not need to be a discrete hard-news
  announcement or one-day occurrence.
- Accept candidates supported by direct article-body or financial-analysis evidence when the exact quote
  states a concrete company-specific or asset-specific fact.
- Accept factual claims inside analysis, opinion, stock-advice, comparison, and listicle articles. Reject
  vague opinion, but do not reject a factual metric or business claim merely because the article is analysis.
- Unknown section text may be accepted only if the quote itself states a concrete company-specific financial,
  market, operating, valuation, guidance, competitive, strategic, or business fact.
- Reject candidates whose only support is boilerplate, publisher promotion, author disclosure, legal disclaimer,
  related links, metadata, footer text, newsletter performance, affiliate compensation, or a recommendation/
  disclosure statement.
- Reject vague qualitative positioning or editorial thesis statements when they lack a concrete company-specific
  action, metric, target, contract, order, reported result, rating, rank, or commitment.
- Reject stock-screener and guru-strategy output even when it is concrete, numeric, and names the company:
  Validea/guru model scores, "rates highest among N guru strategies", "passes all six criteria", "passes all
  tests", "receives a 95% rating under the X strategy", named investor-strategy scorecards, and large-cap or
  mid-cap growth-or-value style classifications. These state a screening algorithm's opinion, not something
  that happened to the company.
- Reject fund and ETF mechanics: expense ratios, assets under management, fund inflows/outflows, top-10
  holdings lists, cost-basis calculations, beta or standard-deviation statistics, and hypothetical
  "$10,000 invested since inception" figures. These describe fund plumbing, not a company event.

Accept when directly quote-supported:
- Reported or estimated revenue, profit, EPS, cash flow, margins, sales, debt, valuation, dividend yield,
  payout, market cap, market share, price movement, subscriber/customer/user counts, production, deliveries,
  bookings, orders, or segment performance.
- Company-specific analyst research outputs inside article evidence, including analyst ratings, upgrades,
  downgrades, price targets, consensus estimates, and analyst ranks when the quote names or clearly refers to
  the company/security. Stock-screener and guru-strategy model output is NOT admissible -- see the reject rules.
- Company guidance, targets, projections, analyst estimates, consensus estimates, price targets, forecasted
  growth, business goals, or future operating plans.
- Market-size, addressable-market, demand, adoption, opportunity, or penetration facts tied to a named company,
  product, service, asset, or market.
- Partnerships, contracts, acquisitions, products, infrastructure, networks, fleet actions, platform activity,
  operational facts, supply/customer relations, or legal/regulatory facts.
- Competitive-position facts such as leadership, dominance, faster growth, market rank, product advantage,
  peer comparison, market-share position, or supplier/customer advantage.

Boilerplate rules:
- Use support_level "boilerplate_only" only when all support comes from publisher promotion, author disclosure,
  legal disclaimer, related links, footer text, metadata, newsletter ads, affiliate compensation, stock-pick
  marketing, or ownership/recommendation disclosure.
- Never use "boilerplate_only" for a candidate whose exact source quote is in core_article_body or
  financial_analysis and directly states a company-specific metric, target, forecast, business fact, market
  fact, competitive fact, or analyst rating.
- Do not call a company-specific analyst rating or rank boilerplate merely because it is produced by a
  research methodology. It is boilerplate only when the quote is promotional, footer, disclosure, or
  advertisement text rather than article evidence. (Screener/guru-strategy scores and fund/ETF mechanics are
  rejected by the reject rules above, not by the boilerplate label.)
- The exact evidence quote is the primary authority. Section labels are helpful context. Section summaries,
  if present, are non-authoritative and must not override a directly supporting quote.

Allowed decision values: accept, reject
Allowed support_level values: direct, article_inferred, unsupported, boilerplate_only

JSON schema:
{{
  "article_id": "{article_id}",
  "candidate_decisions": [
    {{
      "candidate_id": "C1",
      "decision": "reject",
      "support_level": "boilerplate_only",
      "section_type": "publisher_promotion",
      "canonical_source_quote": "",
      "reject_reason": "",
      "confidence": 0.0
    }}
  ]
}}

Article metadata:
article_id: {article_id}
source_ticker: {source_ticker}
article_date: {date}
title: {title}

Candidates to judge:
{candidate_batch_json}
""".strip()


FINAL_CONSOLIDATION_PROMPT_TEMPLATE = """
You are the final event consolidation stage for an evidence-first financial Event Knowledge Graph.
Return one valid JSON object only. Normal formatting with whitespace/indentation is fine -- do not
spend any effort minifying it. Do not include Markdown or prose outside JSON.
Start your response with {{ and end it with }}.

Goal:
- Convert accepted candidates into final event rows.
- Preserve every distinct concrete article-supported event.
- Remove only duplicates, malformed candidates, unsupported claims, and vague commentary.
- Temporal ambiguity is not a reason to remove an accepted candidate. If the event is concrete but time is
  uncertain or belongs to another clause, keep the event and leave temporal_expression blank.
- When accepted candidates come from a sentence with multiple concrete facts, preserve all distinct facts even
  if some have blank temporal_expression.
- Keep separate events when metric concept, company, period, baseline, number, rank, weight, or comparison differs.
- Keep separate events when a sentence contains separate predicates joined by "and", "but", "while", "whereas",
  or similar connectors, especially when one predicate is current/reported and another is future/expected/risk.
- Do not collapse separate facts into a summary event.
- Never merge facts for different companies into one event. Never merge different tickers into one event when
  each ticker has its own price move, valuation, market cap, target price, dividend, metric, or date.
- Never merge different metric families into one event. Split revenue CAGR and EPS CAGR, revenue and profit,
  sales and margins, debt and cash flow, dividend yield and dividend growth, market cap and stock return,
  valuation multiple and guidance, and current metric and future target when each has its own number or period.
- Split stock moves for different securities and split peer-comparison metrics when each company has its own
  metric. Keep comparisons as participants or related_event_mentions only when the comparison itself is the
  single real-world fact.
- Preserve evidence_span_ids, evidence_text, section_type, and support metadata.
- Keep output compact. Do not include reasoning or long prose inside any field. event_description should be one
  concise sentence under 160 characters when possible, evidence_text should remain the shortest exact supporting
  quote, and arrays should contain only directly relevant values/entities.
- financial_metric_mentions must capture BOTH the metric name and its exact value together, one object per
  metric: {{"metric": "dividend_yield", "value": "6.72%", "unit": "%", "time_period": "Q3"}}. Never emit a metric
  name without its value, and never emit a bare number without naming the metric it measures. Preserve the value
  exactly as written, including %, $, currency words, and units. Leave unit or time_period blank when not stated.
- participants and mentioned_entities together must include every company, fund, person, product, or organization
  named in the event_description or evidence_text: participants for entities acting in or directly party to the
  event, mentioned_entities for others merely named. Never drop a named counterparty.
- Populate explicit_causal_relations whenever the evidence states a cause-effect link with wording such as
  "due to", "because", "led to", "drove", "as a result of", "thanks to", or "resulting in". Write each as a short
  "cause -> effect" string, e.g. "biosimilar competition -> Humira sales decline". Leave it empty only when no
  causal wording is present.
- Set importance_score as an integer 1-3 that differentiates events: 3 = material/market-moving (M&A, guidance
  initiation or change, dividend initiation/cut/suspension, major legal/regulatory action, earnings surprise,
  large financing); 2 = notable company-specific result (revenue/profit change, segment or subscriber
  performance, rating/target change, buyback); 1 = routine or background fact (standing yield, P/E, cash, debt
  level, minor commentary). Do not default every event to 1.
- Preserve or correct raw temporal wording in temporal_expression only when it directly modifies the same
  final event fact, metric, or clause. If an accepted candidate carries a time phrase from a neighboring
  fact, remove it by leaving temporal_expression blank.
- The event_description must describe the same predicate that temporal_expression modifies. If the temporal
  expression modifies a future/risk/target/guidance clause, the event_description must describe that future
  clause, not a separate current-state clause.
- Do not put comparison baselines in temporal_expression. A high/low/peak/trough, average, peer, index, or
  prior value is a baseline unless the quote states that it controls the event period.
- Do preserve same-clause reporting or guidance period wording such as full-year, year-to-date, a named year,
  a quarter, a month, or a since-window when it controls the final event's metric, target, forecast, or reported fact.
- Do not reintroduce publisher promotion, author disclosure, legal disclaimer, related-link, footer, or unsupported events.
- Correct event_type when a candidate uses portfolio_holding for a customer order, purchase commitment,
  product delivery, supply/customer relationship, exclusivity clause, or service deal. portfolio_holding is
  only for ownership/allocation/position facts by a portfolio, fund, investor, institution, insider, or author.
- Correct event_type to dividend_announcement for any dividend initiation, raise, cut, suspension, per-share
  amount, payout, or dividend-yield fact that a candidate typed as other.
- Correct event_type away from other whenever a specific type fits: valuation_multiple (P/E, forward earnings
  multiple, price-to-book, EV/EBITDA), balance_sheet (debt levels/maturities, cash, leverage), cash_flow (free
  or operating cash flow, cash burn, runway), analyst_estimate (consensus/analyst estimates, revisions,
  earnings ESP), capital_allocation (capex, investment programs, cost-savings targets), price_level_range
  (52-week high/low, last-trade price), options_activity (puts/calls, strikes, covered calls),
  macroeconomic_event (Fed, PMI, CPI, rates), guidance_update (reiterated/reaffirmed/raised/lowered guidance),
  business_segment_performance (segment revenue/income, gross or operating margin).
- Remove events that are fund/ETF mechanics (expense ratios, AUM, fund flows, holdings, cost basis),
  stock-screener or guru-strategy output (guru scores, "passes all criteria", style classifications), or
  biographical/firm-profile trivia. These are noise, not real-world financial events.

Allowed event_type values: {allowed_types}
Allowed anchor impact_type values: {allowed_impact_types}
Allowed anchor impact_direction values: {allowed_impact_directions}

JSON schema:
{{
  "article_id": "{article_id}",
  "ticker": "{source_ticker}",
  "date": "{date}",
  "events": [
    {{
      "event_type": "",
      "event_trigger": "",
      "event_description": "",
      "main_company": "",
      "ticker": "",
      "participants": [],
      "mentioned_entities": [],
      "financial_metric_mentions": [
        {{"metric": "", "value": "", "unit": "", "time_period": ""}}
      ],
      "temporal_expression": "",
      "explicit_causal_relations": [],
      "related_event_mentions": [],
      "impacted_anchor_companies": [
        {{
          "ticker": "",
          "company": "",
          "impact_type": "",
          "impact_direction": "neutral",
          "evidence": "",
          "raw_mention": ""
        }}
      ],
      "evidence_span_ids": [],
      "evidence_text": "",
      "section_type": "",
      "support_level": "",
      "importance_score": 1,
      "confidence": 0.0
    }}
  ]
}}

Article metadata:
article_id: {article_id}
source_ticker: {source_ticker}
article_date: {date}
title: {title}

Article map:
{article_map_json}

Tracked anchor companies:
{anchors_json}

Accepted candidates:
{accepted_candidates_json}
""".strip()


STAGE6_ANCHOR_IMPACT_PROMPT_TEMPLATE = """
You are the anchor-impact classifier for finalized financial events.
Return one valid JSON object only. Normal formatting with whitespace/indentation is fine -- do not
spend any effort minifying it. Do not include Markdown or prose outside JSON.
Start your response with {{ and end it with }}.

Goal:
- Classify impacted_anchor_companies for the supplied finalized events.
- Do not create, remove, merge, split, rewrite, or reorder events.
- Use the supplied temporary event_id values exactly.
- Return one event_impacts entry for every supplied event_id.
- Use an empty impacted_anchor_companies array when no tracked anchor impact is supported.
- Each impacted_anchor_companies item must contain exactly these keys:
  ticker, impact_type, impact_direction, evidence.
- evidence must be a short quote or phrase from the supplied event fields.

Tracked anchor rules:
- Only classify anchors from tracked_anchor_companies.
- The affected anchor does not need to be the event main_company.
- A tracked anchor can be impacted through a supported business, market, financial, causal,
  portfolio, macro, regulatory, supplier/customer, partner, ecosystem, peer, or competitive relation.
- Do not add an anchor merely because it is mentioned nearby.
- If the event's main company/ticker is already a tracked anchor, include it only when the
  event describes a distinct cross-anchor or anchor-impact relation. Normal main-company
  involvement is represented elsewhere in the graph.

Direction rules:
- impact_direction must be exactly one of positive, negative, or neutral.
- Direction is always from the listed impacted anchor company's perspective.
- Use positive when the event states or strongly implies a benefit, tailwind, demand,
  adoption, pricing power, access, capability, stronger competitive position, share gain,
  revenue/profit opportunity, or market-sentiment benefit for that anchor.
- Use negative when the event states or strongly implies harm, risk, constraint, lost share,
  inferior position, dependency weakness, higher cost, regulatory/legal pressure, demand
  weakness, or market-sentiment harm for that anchor.
- Use neutral only when a real relation exists but benefit/harm is genuinely unclear,
  balanced, indirect, or mixed.

Allowed anchor impact_type values: {allowed_impact_types}
Allowed anchor impact_direction values: {allowed_impact_directions}

JSON schema:
{{
  "article_id": "{article_id}",
  "event_impacts": [
    {{
      "event_id": "E1",
      "impacted_anchor_companies": [
        {{
          "ticker": "",
          "impact_type": "",
          "impact_direction": "",
          "evidence": ""
        }}
      ]
    }}
  ]
}}

Article metadata:
article_id: {article_id}
source_ticker: {source_ticker}
article_date: {date}
title: {title}

Article map:
{article_map_json}

Tracked anchor companies:
{anchors_json}

Final events to classify:
{final_events_json}
""".strip()


TEMPORAL_INVENTORY_PROMPT_TEMPLATE = """
You are temporal Stage T1: temporal mention inventory.
Return one valid JSON object only. Normal formatting with whitespace/indentation is fine -- do not
spend any effort minifying it. Do not include Markdown or prose outside JSON.
Start your response with {{ and end it with }}.

Field-value hygiene (this prevents invalid JSON -- follow it exactly):
- Every string value must be a short literal phrase. Never write full sentences, self-questions,
  deliberation, or step-by-step reasoning inside any value. For "describes" and any reason field, use a
  short noun phrase only (for example "revenue growth" or "stock return"), never an explanation.
- Never put a line break inside a string value, and always close every string with a double quote before
  the next comma, brace, or bracket. Do all of your thinking silently; only the final JSON may appear.

Goal:
- For each finalized event, list all temporal mentions in its event text, temporal_expression,
  evidence_text, and nearby context supplied here.
- Inspect event_description and evidence_text even when temporal_expression is blank. A missing
  temporal_expression does not mean there is no temporal mention.
- Do not normalize dates yet.
- Mark comparison baselines separately from event-time expressions.
- Treat the event's temporal_expression field as a noisy hint, not authoritative evidence. If a phrase appears
  only in temporal_expression but not in the event_description/evidence_text, include it only when the same
  evidence clearly shows it controls this exact event; otherwise mark it as unrelated or omit it.
- A temporal phrase controls an event only when it modifies that same fact, metric, or clause. In compound
  sentences, do not copy a time phrase from one clause/fact to a different clause/fact.
- The temporal mention's described predicate must match the finalized event_description. If the phrase
  describes a different predicate, outcome, risk, target, or expected future effect, mark it unrelated to
  this event even if it appears in the same evidence sentence.
- Classify temporal mentions by their function, not by memorized wording.

Allowed kind values:
calendar_quarter, fiscal_quarter, year, year_to_date, month, day, relative_window,
trailing_52_week, as_of_date, current_state, forecast_period, deadline,
frequency_cadence, comparison_baseline, unknown

JSON schema:
{{
  "article_id": "{article_id}",
  "article_date": "{date}",
  "temporal_inventories": [
    {{
      "event_id": "E1",
      "temporal_mentions": [
        {{
          "text": "",
          "kind": "unknown",
          "explicit_year": "",
          "describes": "",
          "is_comparison_baseline": false,
          "confidence": 0.0
        }}
      ],
      "no_temporal_reason": ""
    }}
  ]
}}

Article metadata:
article_id: {article_id}
source_ticker: {source_ticker}
article_date: {date}
title: {title}

Events:
{events_json}
""".strip()


TEMPORAL_ROLE_SELECTION_PROMPT_TEMPLATE = """
You are temporal Stage T2: temporal role selection.
Return one valid JSON object only. Normal formatting with whitespace/indentation is fine -- do not
spend any effort minifying it. Do not include Markdown or prose outside JSON.
Start your response with {{ and end it with }}.

Field-value hygiene (this prevents invalid JSON -- follow it exactly):
- Every string value must be a short literal phrase. Never write full sentences, self-questions,
  deliberation, or step-by-step reasoning inside any value. For "reason" and similar fields, use a short
  phrase only (for example "controls the metric" or "comparison baseline"), never an explanation.
- Never put a line break inside a string value, and always close every string with a double quote before
  the next comma, brace, or bracket. Do all of your thinking silently; only the final JSON may appear.

Goal:
- Choose the one temporal mention that controls each event's event_time.
- Ignore comparison baselines, valuation baselines, and unrelated dates.
- Do not normalize to ISO dates yet.
- If no mention safely controls event time, return selected_temporal_text as empty and role unknown.
- Hard rule: frequency_cadence mentions cannot be selected as event_time. Put cadence-only phrases in
  ignored_temporal_mentions and return selected_temporal_text "", role unknown, time_kind unknown unless
  there is a separate concrete calendar anchor.
- Hard rule: valuation-basis phrases are NOT event times. "forward earnings", "expected earnings",
  "trailing earnings", "forward P/E", "next twelve months earnings", and similar earnings-basis wording
  describe how a multiple is measured, not when the event happened. Put them in ignored_temporal_mentions
  and return selected_temporal_text "", role unknown, time_kind unknown unless a separate concrete calendar
  anchor controls the fact.
- Use this reasoning order: classify the expression kind, decide the event_time role, identify the anchor
  date or period, separate controlling time from comparison/baseline text, then choose the controlling phrase.
- The controlling phrase must belong to the event's own fact, metric, or clause. In a compound sentence with
  two facts, a time phrase attached to fact A must not be selected for fact B.
- The chosen temporal mention's described predicate must match the event_description. If the only available
  temporal mention describes a different predicate, outcome, risk, target, or expected future effect, select
  no temporal text for this event and return role unknown, or current/as-of only when the event text supports it.
- In sentences with a current/reported clause plus a separate future modal clause, a future period controls only
  the future modal clause. It must not be selected for the current/reported clause.
- A present-tense or reported deterioration/pressure/change fact is not automatically a future forecast just
  because the same sentence later mentions a future period for a different possible consequence.
- Treat the event's temporal_expression field as a noisy hint. Do not select it merely because it is supplied.
  Prefer temporal text that appears in the event_description or evidence_text and controls that exact fact.
  If the supplied temporal_expression does not appear in those fields and no same-clause support is present,
  ignore it.
- If temporal_expression is blank, still inspect event_description and evidence_text for a same-clause period.
  Do not return unknown merely because temporal_expression is blank.

Allowed event_time role values:
occurrence_date, reported_period, measurement_window, guidance_period, deadline,
future_effective_period, announcement_date, historical_event, unknown

Allowed time_kind values:
calendar_quarter, fiscal_quarter, year_to_date, relative_window, trailing_52_week,
as_of_date, current_state, point_in_time_metric, forecast_period, deadline,
frequency_cadence, unknown

JSON schema:
{{
  "article_id": "{article_id}",
  "article_date": "{date}",
  "temporal_roles": [
    {{
      "event_id": "E1",
      "selected_temporal_text": "",
      "role": "unknown",
      "time_kind": "unknown",
      "ignored_temporal_mentions": [
        {{"text": "", "reason": ""}}
      ],
      "reason": "",
      "confidence": 0.0
    }}
  ]
}}

Article metadata:
article_id: {article_id}
source_ticker: {source_ticker}
article_date: {date}
title: {title}

Events:
{events_json}

Temporal inventories:
{inventories_json}
""".strip()


TEMPORAL_NORMALIZATION_EVIDENCE_PROMPT_TEMPLATE = """
You are temporal Stage T3: normalize selected temporal expressions.
Return one valid JSON object only. Normal formatting with whitespace/indentation is fine -- do not
spend any effort minifying it. Do not include Markdown or prose outside JSON.
Start your response with {{ and end it with }}.

Field-value hygiene (this prevents invalid JSON -- follow it exactly):
- Every string value must be a short literal phrase. Never write full sentences, self-questions,
  deliberation, or step-by-step reasoning inside any value. source_text must be the exact short temporal
  phrase only, never an explanation.
- Never put a line break inside a string value, and always close every string with a double quote before
  the next comma, brace, or bracket. Do all of your thinking silently; only the final JSON may appear.

Goal:
- Convert only the selected temporal expression from T2 into structured event_time.
- Use article_date only as reference context for relative or incomplete time expressions.
- If selected_temporal_text is empty or unsafe, output role unknown with blank dates.
- For periods, output full start and end dates. Do not collapse quarters, years, year-to-date ranges,
  trailing windows, or multi-period measurement windows to one day.
- For any expression classified by T2 as a calendar_quarter, use fixed calendar months independent of
  article_date.
  Article_date may supply the year, but it must never change which months belong to the quarter number.
- Calendar quarter to month mapping (always use these four ranges; never derive quarter months from
  article_date or from which quarter the article itself was published in). The year below is illustrative
  only (2023) -- substitute the actual year chosen per the year-inference rules above:
  Q1 = January-March, e.g. 2023-01-01 to 2023-03-31
  Q2 = April-June, e.g. 2023-04-01 to 2023-06-30
  Q3 = July-September, e.g. 2023-07-01 to 2023-09-30
  Q4 = October-December, e.g. 2023-10-01 to 2023-12-31
  A quarter's months never shift because the article itself was published in a different quarter.
- If a quarter phrase omits the year, use the article_date year unless the same selected expression or same
  event evidence explicitly names a different year. Do not borrow years from neighboring events or sentences.
- Do not reinterpret an ordinal quarter as the current article-date quarter, the next quarter, or the most
  recent completed quarter. Normalize the quarter named in the selected expression itself.
- Apply quarter mapping only to explicit ordinal-quarter wording. Vague part-of-year wording is not a quarter
  expression; normalize it as an approximate range with granularity range, or mark unknown if too vague.
- Fiscal quarters must not be converted to calendar quarters unless exact fiscal calendar dates are supplied.
- If year is omitted, infer from article_date and article context. Do not infer a future year merely because the
  event is a forecast, guidance, target, deadline, or projection. A future year requires explicit next-year,
  future-year, or different fiscal/reporting-year wording.
- Relative-year phrases mean the article year unless the evidence explicitly names next year, a future year,
  or a different fiscal/reporting year.
- Year-to-date style phrases mean Jan 1 of the article year through article_date unless a different year is explicit.
- Since-style cumulative changes mean the range from the named start date/year/event through article_date/current
  context unless the evidence supplies a different end date. Do not collapse "since YEAR" to only that year.
- Contract-duration phrases describe duration. Do not turn an N-year duration into the event occurrence date.
  Use the duration as the event_time range only when the event is explicitly about the effective contract period.
- Frequency or cadence phrases such as annually, quarterly, monthly, recurring, every year, or every N years
  are not event_time ranges by themselves and must not be converted to the next calendar year/quarter/month.
  If T2 selected only a cadence phrase or time_kind frequency_cadence with no separate concrete calendar
  anchor, return exactly: role unknown, start_date "", end_date "", granularity unknown, source_text "",
  confidence 0.0.
- Non-future facts must not end after article_date.

Allowed role values:
occurrence_date, reported_period, measurement_window, guidance_period, deadline,
future_effective_period, announcement_date, historical_event, unknown

Allowed granularity values:
day, month, quarter, year, range, unknown

JSON schema:
{{
  "article_id": "{article_id}",
  "article_date": "{date}",
  "event_times": [
    {{
      "event_id": "E1",
      "event_time": {{
        "role": "unknown",
        "start_date": "",
        "end_date": "",
        "granularity": "unknown",
        "source_text": "",
        "confidence": 0.0
      }}
    }}
  ]
}}

Article metadata:
article_id: {article_id}
source_ticker: {source_ticker}
article_date: {date}
title: {title}

Events:
{events_json}

Temporal roles:
{roles_json}
""".strip()


TEMPORAL_JUDGE_PROMPT_TEMPLATE = """
You are temporal Stage T4: a mechanical JSON verifier.
Return one valid JSON object only. Normal formatting with whitespace/indentation is fine -- do not
spend any effort minifying it. Do not include Markdown or prose outside JSON.
Start your response with {{ and end it with }}.
Do not reason step by step. Do not debate alternatives. Reasons must be short codes or short phrases.

Field-value hygiene (this prevents invalid JSON -- follow it exactly):
- Every string value must be a short literal phrase. Never write full sentences, self-questions,
  deliberation, or step-by-step reasoning inside any value. The reason field must be a short code or phrase
  (for example "wrong_quarter" or "accept"), never an explanation.
- Never put a line break inside a string value, and always close every string with a double quote before
  the next comma, brace, or bracket. Do all of your thinking silently; only the final JSON may appear.

Task:
- For each event, audit the supplied normalized event_time.
- Return exactly one judgment per event_id.
- Do not create, remove, merge, split, or rewrite events.

Output policy:
- Use decision "accept" only when current_event_time matches the event evidence and temporal role.
- Use decision "correct" when replacing dates, role, granularity, source_text, or confidence.
- Use decision "unknown" when time cannot be safely normalized.
- Keep reason under 8 words.

Calendar quarter reference for catching wrong_quarter errors (year is illustrative only, substitute the
event's actual year): Q1 = 2023-01-01 to 2023-03-31. Q2 = 2023-04-01 to 2023-06-30.
Q3 = 2023-07-01 to 2023-09-30. Q4 = 2023-10-01 to 2023-12-31. A quarter's months never shift because
the article itself was published in a different quarter than the one it reports on.

Allowed decision values: accept, correct, unknown
Allowed error_type values:
none, wrong_quarter, wrong_year, comparison_baseline_used, unsafe_fiscal_calendar,
article_date_overused, unsupported_time, other

JSON schema:
{{
  "article_id": "{article_id}",
  "article_date": "{date}",
  "temporal_judgments": [
    {{
      "event_id": "E1",
      "decision": "accept",
      "error_type": "none",
      "event_time": {{
        "role": "unknown",
        "start_date": "",
        "end_date": "",
        "granularity": "unknown",
        "source_text": "",
        "confidence": 0.0
      }},
      "reason": ""
    }}
  ]
}}

Article metadata:
article_id: {article_id}
source_ticker: {source_ticker}
article_date: {date}
title: {title}

Events:
{events_json}

Temporal inventories:
{inventories_json}

Temporal roles:
{roles_json}

Normalized event times to audit:
{event_times_json}
""".strip()
