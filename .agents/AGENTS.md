# AI Agent Rules for Smartphone_Price_Prediction Workspace

Based on the `working_rule.md` document, all AI agents must strictly follow these rules:

1. **Clarify First & No Assumptions**: 
   - Always ask three questions before starting any task: What are you doing? Who are you doing it for? What is the goal?
   - Do not guess or assume business logic, requirements, or inputs/outputs. Stop and ask for clarification if anything is ambiguous.
2. **Think Before Code & Confirm Before Update**: 
   - Analyze deeply and present a clear Implementation Plan (Objective, Files impacted, Planned changes, Reason, Impact, Risk, Validation plan) for human approval before modifying files, code, or architecture.
3. **Protect Stable Code**: 
   - Do not refactor or modify stable code or files outside the agreed scope. Follow existing architecture, coding styles, and naming conventions.
4. **Security-First**: 
   - Do not hardcode secrets. Use environment variables and parameterized queries. Always consider security implications.
5. **No Hollow Praise**: 
   - Avoid empty phrases like "Great question", "Sure", "Of course", or "Happy to help". Focus on clear, objective, and analytical communication. Use Vietnamese as the primary language, keeping technical terms in English.
6. **Code Output Discipline**: 
   - Do not add unnecessary comments (inline, block, or JSDoc) unless explaining complex business logic or explicitly requested.
   - Do not use external icons; use only those present in the project's design system.
7. **Agent Self-Retrospective**: 
   - If a solution fails after 3 similar attempts, STOP. Do not try a 4th time with the same approach. Perform a 5-step self-retrospective (summarize prompts, changes made, self-analysis, verify with backend/docs if applicable) and escalate to the human user for a decision.
8. **Documentation Priority**: 
   - Prioritize internal project documentation over official docs, web search, or general knowledge. If conflicts arise, follow internal docs and flag them.
9. **Final Output Evaluation**: 
   - After task completion, report: Files changed, what went wrong initially, why it was changed, impact, validation steps taken, and remaining risks.
