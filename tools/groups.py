"""Noto file code -> Unicode group, subgroup, name and the emoji itself (emoji-test.txt). Writes codes_meta.json."""
import json, re
meta, group, sub = {}, None, None
for line in open("emoji-test.txt", encoding="utf-8"):
    if line.startswith("# group:"):
        group = line.split(":", 1)[1].strip(); continue
    if line.startswith("# subgroup:"):
        sub = line.split(":", 1)[1].strip(); continue
    m = re.match(r"^([0-9A-F ]+)\s*;\s*(fully-qualified|component)\s*#\s*(\S+)\s+E[\d.]+\s+(.*)$", line)
    if not m:
        continue
    cps = [c.lower() for c in m.group(1).split() if c.lower() != "fe0f"]
    code = "_".join(cps)
    meta.setdefault(code, {"group": group, "subgroup": sub, "name": m.group(4).strip(), "char": m.group(3)})
codes = [l.strip() for l in open("all_codes.txt") if l.strip()]
out, miss = {}, []
def extra(c):
    """Noto files with no fully-qualified emoji-test.txt line: lone regional indicators, keycap parts, an old
    sequence order (the blond-woman tones are 1f471_1f3fb_200d_2640 in Unicode) and Noto's private-use fe82b."""
    cp = c.split("_")
    if len(cp) == 1 and 0x1F1E6 <= int(cp[0], 16) <= 0x1F1FF:
        L = chr(ord("A") + int(cp[0], 16) - 0x1F1E6)
        return {"group": "Symbols", "subgroup": "regional-indicator", "name": f"regional indicator {L}", "char": chr(int(cp[0], 16))}
    if c in ("0023", "002a", "20e3"):
        nm = {"0023": "number sign", "002a": "asterisk", "20e3": "keycap"}[c]
        return {"group": "Symbols", "subgroup": "keycap", "name": nm, "char": chr(int(c, 16))}
    if len(cp) == 4 and cp[2] == "2640" and cp[1] == "200d":
        alt = "_".join([cp[0], cp[3], cp[1], cp[2]])
        if alt in meta:
            return dict(meta[alt])
    if c == "fe82b":
        return {"group": "Symbols", "subgroup": "other", "name": "unknown (Noto private use)", "char": ""}
    return None


for c in codes:
    m = meta.get(c) or extra(c)
    if m:
        out[c] = m
    else:
        miss.append(c)
json.dump(out, open("codes_meta.json", "w"), ensure_ascii=False, indent=0)
from collections import Counter
print(len(out), "matched;", len(miss), "unmatched:", miss[:12])
print(Counter(v["group"] for v in out.values()).most_common())
