You are a senior AI engineer designing a production-grade agentic AI system
for graduate-level experimental data analysis and academic research support.

The system MUST be implemented using LangGraph.

========================
SYSTEM GOAL
========================
Build a LangGraph-based agentic AI system that:

1. Accepts experimental data and experiment context
2. Produces statistically valid analysis results
3. Generates academically sound scientific interpretation
4. Handles advisor/reviewer feedback through a rebuttal mechanism
5. Iteratively refines analysis or interpretation when required

This system must mirror real academic research workflows,
including advisor critique and revision cycles.

========================
CORE DESIGN CONSTRAINTS
========================
- Python-based
- LangGraph StateGraph architecture REQUIRED
- Each agent is a graph node (NO monolithic agent)
- Agents must be logically stateless
- Shared state must flow ONLY through LangGraph state
- All intermediate outputs must be persisted as artifacts (JSON/YAML/MD)
- Partial re-execution must be supported via graph edges
- Never fabricate results or statistical evidence

========================
GLOBAL GRAPH STATE
========================
Define a TypedDict-based shared state, including but not limited to:

- experiment_context
- data_schema
- data_quality_report
- analysis_plan
- execution_results
- report_text
- reviewer_comment (optional)
- rebuttal_decision (optional)
- artifacts_path

========================
REQUIRED GRAPH NODES (AGENTS)
========================

1. ContextAgentNode
-------------------
Responsibility:
- Parse experiment description
- Extract objective, hypothesis, variables, constraints

Input State:
- raw_experiment_description

Output State:
- experiment_context

Persist:
- context.json

2. DataUnderstandingAgentNode
-----------------------------
Responsibility:
- Load experimental data
- Infer schema and data types
- Detect missing values, outliers, unit issues

Input State:
- experiment_context
- raw_experimental_data

Output State:
- data_schema
- data_quality_report

Persist:
- schema.yaml
- data_quality_report.md

3. AnalysisPlanningAgentNode
----------------------------
Responsibility:
- Decide analysis steps and methods
- Explicitly define statistical assumptions
- Define validation strategy

Input State:
- experiment_context
- data_schema
- data_quality_report

Output State:
- analysis_plan

Persist:
- analysis_plan.yaml

4. ExecutionAgentNode
---------------------
Responsibility:
- Generate Python analysis code
- Execute analysis robustly
- Produce figures, tables, metrics

Input State:
- analysis_plan
- raw_experimental_data

Output State:
- execution_results

Persist:
- results.json
- figures/
- execution_log.txt

5. ReportingAgentNode
---------------------
Responsibility:
- Interpret results conservatively
- Generate publication-ready academic prose
- Explicitly state assumptions and limitations

Input State:
- experiment_context
- analysis_plan
- execution_results

Output State:
- report_text

Persist:
- report.md

6. RebuttalAgentNode
--------------------
Responsibility:
- Process advisor/reviewer comments
- Classify criticism type and severity
- Decide response strategy
- Request reanalysis if required
- Generate academically valid rebuttal text

IMPORTANT:
- This node MUST NOT access raw experimental data
- It operates ONLY on prior artifacts and review comments

Input State:
- reviewer_comment
- experiment_context
- analysis_plan
- execution_results
- report_text

Output State:
- rebuttal_decision
- rebuttal_response_text

Persist:
- rebuttal_decision.yaml
- rebuttal_response.md

========================
GRAPH CONTROL FLOW
========================
Implement a LangGraph StateGraph with:

Primary Path:
START
 → ContextAgentNode
 → DataUnderstandingAgentNode
 → AnalysisPlanningAgentNode
 → ExecutionAgentNode
 → ReportingAgentNode
 → END

Conditional Review Path:
If reviewer_comment exists:
 ReportingAgentNode
   → RebuttalAgentNode
     → Conditional Edge:
        - explain_only → END
        - concede_and_limit → ReportingAgentNode (update interpretation only)
        - reanalyze → AnalysisPlanningAgentNode → ExecutionAgentNode → ReportingAgentNode → END

Graph MUST support:
- Conditional edges
- Re-entry into prior nodes
- Preservation of previous artifacts (no overwrite)

========================
DELIVERABLES
========================
Generate:

1. LangGraph StateGraph construction code
2. TypedDict definition for shared state
3. Node (agent) function skeletons
4. Conditional edge logic
5. Artifact persistence utilities
6. Clear docstrings explaining:
   - Agent responsibilities
   - State inputs/outputs
   - Re-entry conditions

========================
DO NOT
========================
- Implement agents as a single chain
- Hide statistical assumptions
- Overstate significance
- Use AutoML abstractions
- Bypass LangGraph control flow

Tone of generated reports and rebuttals:
Professional, restrained, academically conservative.
