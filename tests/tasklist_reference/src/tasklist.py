"""Reference solution for the ag_module agent task.

Exists so the grader can be trusted: if a correct implementation does not score
every check, the fault is in the checks rather than in the model being measured.
"""
import json

FIELDS = ("title", "description", "color", "done")


def new_store():
    return {"tasks": [], "next_id": 1}


def add(store, title, description="", color="#888888"):
    if not title:
        raise ValueError("title is required")
    task = dict(id=store["next_id"], title=title, description=description,
                color=color, done=False, position=len(store["tasks"]))
    store["next_id"] += 1
    store["tasks"].append(task)
    return task


def get(store, task_id):
    for task in store["tasks"]:
        if task["id"] == task_id:
            return task
    return None


def update(store, task_id, **fields):
    unknown = [k for k in fields if k not in FIELDS]
    if unknown:
        raise ValueError(f"cannot set {', '.join(sorted(unknown))}")
    task = get(store, task_id)
    if task is None:
        return None
    task.update(fields)
    return task


def delete(store, task_id):
    task = get(store, task_id)
    if task is None:
        return False
    store["tasks"].remove(task)
    return True


def tasks(store, done=None, q=None):
    out = sorted(store["tasks"], key=lambda t: t["position"])
    if done is not None:
        out = [t for t in out if bool(t["done"]) is bool(done)]
    if q:
        needle = q.lower()
        out = [t for t in out
               if needle in t["title"].lower() or needle in t["description"].lower()]
    return out


def reorder(store, order):
    known = {t["id"] for t in store["tasks"]}
    unknown = [i for i in order if i not in known]
    if unknown:
        raise ValueError(f"no such task: {unknown[0]}")
    for position, task_id in enumerate(order):
        get(store, task_id)["position"] = position
    return tasks(store)


def save(store, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(store, fh)


def load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return new_store()
