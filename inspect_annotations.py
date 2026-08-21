"""
Quick diagnostic — figures out whether annotations.json is a Label Studio
export (list of tasks) or a COCO export (dict with images/annotations/
categories), and shows why load_annotations() parsed 0 images either way.
Run from the project root: python inspect_annotations.py
"""
import json
from pathlib import Path

path = Path("data/gold_standard/annotations/annotations.json")
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 72)
print(f"  File: {path}")
print("=" * 72)

# ── Format detection ──────────────────────────────────────────────────────
if isinstance(data, dict):
    top_keys = set(data.keys())
    coco_keys = {"images", "annotations", "categories"}
    if coco_keys.issubset(top_keys):
        print("\n  VERDICT: This is a COCO-format export, NOT Label Studio JSON.")
        print(f"  Top-level keys: {sorted(top_keys)}")
        print(f"  'images'      : {len(data.get('images', []))} entries")
        print(f"  'annotations' : {len(data.get('annotations', []))} entries")
        print(f"  'categories'  : {[c.get('name') for c in data.get('categories', [])]}")
        print("\n  This matches the CVAT/COCO 1.0 format used for")
        print("  symptom_annotations.json (the SEPARATE symptom-region pass),")
        print("  not the Label Studio polygonlabels export load_annotations()")
        print("  expects for the leaf-silhouette pass. If this is genuinely")
        print("  your leaf-silhouette annotation file, it was likely exported")
        print("  from the wrong tool/project, or symptom_annotations.json got")
        print("  copied/renamed to annotations.json by mistake.")
        if data.get("images"):
            print("\n  First image entry:", json.dumps(data["images"][0], indent=2)[:500])
        if data.get("annotations"):
            print("\n  First annotation entry:", json.dumps(data["annotations"][0], indent=2)[:500])
        raise SystemExit
    else:
        print(f"\n  Single dict, not a list — wrapping as [data]. Top-level keys: {sorted(top_keys)}")
        data = [data]

if not isinstance(data, list):
    print(f"\n  Unexpected top-level type: {type(data)} — not a list or dict.")
    raise SystemExit

print(f"\n  Top-level type: list")
print(f"  Number of tasks: {len(data)}")

if not data:
    print("\n  VERDICT: The file is an empty list — nothing was exported at all.")
    raise SystemExit

task = data[0]
if not isinstance(task, dict):
    print(f"\n  VERDICT: List items aren't dicts (got {type(task)}) — unrecognized format.")
    raise SystemExit

print("\n  First task top-level keys:", list(task.keys()))

if "annotations" in task:
    anns = task["annotations"]
    print(f"  task['annotations'] has {len(anns)} entries")
    if anns:
        ann = anns[0]
        print("  First annotation keys:", list(ann.keys()))
        if "result" in ann:
            result = ann["result"]
            print(f"    result has {len(result)} entries")
            if result:
                r = result[0]
                print("    First result keys:", list(r.keys()))
                print("    First result 'type':", repr(r.get("type")))
                print("    First result 'value' keys:", list(r.get("value", {}).keys()))
                print("\n  VERDICT: This looks like a valid Label Studio JSON export.")
                print("  If load_annotations() still parsed 0, check whether 'type'")
                print("  above says 'polygonlabels' exactly — if it says something")
                print("  else (e.g. 'polygon', 'rectanglelabels'), that's the mismatch.")
            else:
                print("\n  VERDICT: 'result' is empty — this task has an annotation")
                print("  record but no actual drawn regions in it (skipped/empty task).")
        else:
            print("\n  VERDICT: annotation object has no 'result' key — check Label")
            print("  Studio export version/settings.")
    else:
        print("\n  VERDICT: 'annotations' list is empty for this task — task was")
        print("  exported before being annotated, or export included")
        print("  unannotated/skipped tasks only.")
else:
    print("\n  VERDICT: No 'annotations' key on this task at all.")
    print("  This is almost certainly a JSON-MIN export, not full JSON —")
    print("  JSON-MIN flattens label results directly onto the task instead")
    print("  of nesting them under annotations -> result.")
    print("\n  Full first task (truncated to 1500 chars):")
    print(json.dumps(task, indent=2)[:1500])
