#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/supp10_utility.log
note() { echo "[$(date +%T)] $*" >> $LOG; }
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY HF_HUB_OFFLINE=1
if curl -s localhost:8000/v1/models > /dev/null; then note "port 8000 already serving: abort"; exit 1; fi
nohup serve/vllm_qwen3_14b.sh > logs/vllm_supp10.log 2>&1 < /dev/null &
VLLM=$!
until curl -s localhost:8000/v1/models > /dev/null; do
  kill -0 $VLLM 2> /dev/null || { note "vLLM exited"; exit 1; }; sleep 10
done
WS=()
for p in 3000 3001 3002 3003 3004; do
  ss -ltn | grep -q ":$p " && { note "port $p busy: abort"; kill $VLLM; exit 1; }
  bash scripts/webshop_instance.sh $p >> $LOG
  WS+=($(cat logs/webshop_server_$p.pid))
done
note "vLLM $VLLM and WebShop ${WS[*]} ready"
stop() { kill $VLLM ${WS[*]}; note "$1 (servers stopped)"; exit 1; }
python3 - <<'EOF' >> $LOG 2>&1 || stop "goal ids do not match the instructions"
import html, json, re, urllib.request
w = json.load(open("baselines/MINJA/rap/webshop_instructions.json"))
norm = lambda s: re.sub(r"price lower than [0-9.]+ dollars", "", re.sub(r"\s+", " ", html.unescape(s)).strip().lower())
for n in (1, 2, 3, 777, 2024, 5000, 8191, 10586):
    page = urllib.request.urlopen(f"http://localhost:3000/fixed_0_{n}").read().decode()
    assert norm(w[n]["instruction"]) in norm(page), (n, w[n]["instruction"])
print("goal ids match (apart from price limits)")
EOF
bash scripts/supp10_utility_job.sh 0 0 recovered_contain
python3 - <<'EOF' >> $LOG 2>&1 || stop "smoke check failed"
import json
v = json.load(open("baselines/MINJA/rap/victim_target_pair/victim_target.json"))[0]["victim"]
f = f"results/minja_recovery/qwen3-14b-main/{v.replace(' ', '_')}/seed0/utility_recovered_contain.json"
u = json.load(open(f))
s, sc = u["summary"], [e["score"] for e in u["episodes"]]
assert s["victim"]["n"] == 20 and s["other"]["n"] == 10, s
assert any(0 < x < 1 for x in sc) or any(x == 1 for x in sc), sc
print("smoke", f, s)
EOF
{ for s in 0 1 2; do for i in 0 1 2 3 4 5 6 7 8; do echo "$i $s"; done; done; } \
  | xargs -P 5 -L 1 bash scripts/supp10_utility_job.sh
kill $VLLM ${WS[*]}
note "SUPP10_UTILITY_DONE (servers stopped)"
python3 scripts/utility_summary.py >> $LOG 2>&1
