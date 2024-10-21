import json
import sys

def is_string(s):
  return isinstance(s, str)

fls = [s.strip() for s in open("file_list.txt") if not s.startswith('#')]

fout = open("text_reports.txt", "w")

for fn in fls:
  f = open(fn)
  a = json.load(f)
  f.close()

  for e in a:
    Findings2 = e["text"]["Findings2"]
    Prompt2 = e["text"]["Prompt2"]
    if not is_string(Findings2):continue
    if not is_string(Prompt2):continue

    s = f"Findings2: {Findings2} Prompt2: {Prompt2}"

    print(s.strip(), file=fout)

  sys.stdout.flush()

fout.close()

print('Finished.')
