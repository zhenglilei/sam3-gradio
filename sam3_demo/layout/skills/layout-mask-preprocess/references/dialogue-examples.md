# Dialogue examples

The following examples demonstrate response shape only. Candidate IDs must come from the current request.

User: `Analyze automatically`

```json
{"schema_version":1,"intent":"analyze","profile":"ACT","confidence":0.91,"selected_candidate_id":"C2","observations":["Many small edge concavities are visible"],"assistant_message":"Select C2 to increase close while preserving line width.","manual_review":false,"warnings":[]}
```

User: `Fill a little more without thickening`

```json
{"schema_version":1,"intent":"revise","profile":"ACT","confidence":0.88,"selected_candidate_id":"C3","observations":["C3 fills slightly more than the current draft while preserving outer width"],"assistant_message":"Select C3; it changes only close.","manual_review":false,"warnings":[]}
```

User: `Preserve holes`

```json
{"schema_version":1,"intent":"revise","profile":"GE2","confidence":0.82,"selected_candidate_id":"C1","observations":["C1 preserves square holes best"],"assistant_message":"Keep C1 and review the yellow risk border.","manual_review":true,"warnings":["Local edge differences are visible"]}
```
