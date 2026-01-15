# Gemini Agent Notes

This folder mirrors Ralph's build flow so Gemini CLI can follow the same technique without installing Ralph.

Inputs:
- `.agents/tasks/prd.json` (Ralph PRD JSON)
- `.agents/ralph/references/GUARDRAILS.md`
- `.agents/ralph/references/CONTEXT_ENGINEERING.md`
- `.agents/gemini/PROMPT_build.md` (this prompt)

Recommended usage (PowerShell):
```
Get-Content C:\users\lenovo\projects\mcc_class\.agents\gemini\PROMPT_build.md | gemini -o text
```

Note: This Gemini CLI build does not accept a `--search` flag, so run without it.

Notes:
- Gemini should read the PRD JSON and work one story at a time.
- Keep edits scoped and follow the guardrails.
