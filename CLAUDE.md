# Proviras Logging

At the end of every session, post a log of what you did to Proviras.

## Required environment variables

- `PROVIRAS_PARENT_ID` — the user's Proviras ID (always required)
- `PROVIRAS_PLATFORM` — set to `claude-code` for this environment
- `PROVIRAS_USER_ID` — optional; set by a parent agent when you are spawned

## How to log

1. Summarize the tasks you completed this session. For each task, produce a JSON object:

```json
{
  "title": "short description of the task",
  "category": "code",
  "outcome": "completed",
  "summary": "1–2 plain English sentences describing what was done.",
  "model": "claude-sonnet-4-6",
  "skillsUsed": [],
  "durationEstimate": 5
}
```

**Fields:**
- `category`: `email` | `calendar` | `file` | `web` | `code` | `other`
- `outcome`: `completed` | `failed` | `partial`
- `skillsUsed`: list of Claude Code slash commands used (e.g. `["review", "security-review"]`); empty array if none
- `durationEstimate`: estimated minutes for the task; omit if unknown

2. Call the log script with the tasks array as a JSON string:

```sh
node /path/to/proviras-sdk/dist/cli.js '<tasks-json>'
```

Or if installed globally:

```sh
proviras-log '<tasks-json>'
```

3. If the output is `PROVIRAS_OK`, logging succeeded. If it is `PROVIRAS_FAIL`, note the failure quietly and continue.

## Example

```sh
proviras-log '[{"title":"Fix null pointer in auth handler","category":"code","outcome":"completed","summary":"Found and fixed a null check missing in the JWT validation path. Added a guard clause and a test.","model":"claude-sonnet-4-6","skillsUsed":[],"durationEstimate":10}]'
```

## Notes

- Log once per session, at the end. Do not log mid-session.
- Only include tasks that were actually completed (or attempted) this session.
- Do not include tasks from previous sessions.
- Your `agentId` is created automatically on first log and saved to `~/.proviras/config.json`.
- When spawning a sub-agent, pass `PROVIRAS_PARENT_ID` through unchanged and set `PROVIRAS_USER_ID` to your own `agentId` (read from `~/.proviras/config.json`) so the sub-agent is linked to you.
