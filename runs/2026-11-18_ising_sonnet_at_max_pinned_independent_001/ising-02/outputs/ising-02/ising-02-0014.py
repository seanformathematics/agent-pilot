
import json
p = json.load(open("/work/submission_payload.json"))
print(json.dumps(p["seeds"], separators=(",", ":")))
