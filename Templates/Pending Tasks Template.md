---
type: task-dashboard
created: {{date:YYYY-MM-DD}}
tags:
  - dashboard/tasks
---

# 📋 Pending Tasks Action Center

> [!TIP]
> This note dynamically tracks and filters all uncompleted tasks across your vault using Dataview.

---

## 🔴 Overdue Tasks
```dataview
TASK
FROM ""
WHERE !completed AND due AND due < date(today)
SORT due ASC
```

---

## 🟡 Due Today & Next 2 Days
```dataview
TASK
FROM ""
WHERE !completed AND due AND due >= date(today) AND due <= date(today) + dur(2 days)
SORT due ASC
```

---

## ⚪ Upcoming & Backlog Tasks
```dataview
TASK
FROM ""
WHERE !completed AND (!due OR due > date(today) + dur(2 days))
SORT file.mtime DESC
```
