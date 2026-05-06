#!/usr/bin/env node
import { ProvirasSdk, Task } from "./index";

void (async () => {
  const [,, tasksArg, periodStartArg] = process.argv;

  if (!tasksArg) {
    console.error("Usage: proviras-log '<tasks-json>' [period-start-iso]");
    process.exit(1);
  }

  let tasks: Task[];
  try {
    tasks = JSON.parse(tasksArg);
  } catch {
    console.error("Invalid tasks JSON");
    process.exit(1);
  }

  const periodStart = periodStartArg ? new Date(periodStartArg) : undefined;
  const sdk = new ProvirasSdk();
  const session = sdk.startSession(periodStart);
  tasks.forEach((t) => session.addTask(t));
  const ok = await session.end();

  if (ok) {
    console.log("PROVIRAS_OK");
    process.exit(0);
  } else {
    console.error("PROVIRAS_FAIL");
    process.exit(1);
  }
})();
