#!/usr/bin/env python3
"""Renew the paper-reader S3 backup key via the DexGEM oss-bot (Feishu).

Runs on the workstation (needs lark-cli user auth + ssh dexgem access).
Flow: @bot in 算力-DexGEM with a 168h request (auto-approved for runtime),
poll the bot's p2p chat for the delivery message, install into
/etc/paper-reader/s3-backup.env on the VPS. Safe to run weekly (e.g. cron).
Never prints the secret.
"""
import json, re, subprocess, sys, time

BOT_OPEN_ID = "ou_23025a174ae102ee12b128d83fcc6b1e"  # 陈思远的飞书 CLI (oss-bot)
GROUP = "oc_ba2f4775ff7652413e0a239a8fdbe876"        # 🖥 算力-DexGEM
BOT_P2P_CHAT = "oc_bc151b8db6b018a16aebe256d2c25fcd"  # bot's key-delivery p2p

REMOTE_INSTALL = '''
import json, sys
creds = json.load(sys.stdin)
p = "/etc/paper-reader/s3-backup.env"
lines = open(p).read().splitlines()
out = []
for ln in lines:
    if ln.startswith("AWS_ACCESS_KEY_ID="):
        ln = "AWS_ACCESS_KEY_ID=" + creds["ak"]
    elif ln.startswith("AWS_SECRET_ACCESS_KEY="):
        ln = "AWS_SECRET_ACCESS_KEY=" + creds["sk"]
    out.append(ln)
open(p, "w").write("\\n".join(out) + "\\n")
print("installed new key ending", creds["ak"][-4:])
'''

def lark(method, path, params=None, data=None):
    cmd = ["lark-cli", "api", method, path, "--as", "user", "--format", "json"]
    if params: cmd += ["--params", json.dumps(params, ensure_ascii=False)]
    if data: cmd += ["--data", json.dumps(data, ensure_ascii=False)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return json.loads(out.stdout)

def main():
    before = int(time.time() * 1000)
    text = (f'<at user_id="{BOT_OPEN_ID}"></at> oss-key /runtime/paper-reader/backups '
            "168h paper-reader 每日快照备份自动续期 (reader.dex-gem.ai)")
    r = lark("POST", "/open-apis/im/v1/messages", {"receive_id_type": "chat_id"},
             {"receive_id": GROUP, "msg_type": "text",
              "content": json.dumps({"text": text}, ensure_ascii=False)})
    if not r.get("ok"):
        sys.exit(f"request send failed: {r.get('error')}")
    print("request sent; waiting for delivery...")
    deadline = time.time() + 300
    while time.time() < deadline:
        time.sleep(20)
        r = lark("GET", "/open-apis/im/v1/messages",
                 {"container_id_type": "chat", "container_id": BOT_P2P_CHAT,
                  "page_size": 5, "sort_type": "ByCreateTimeDesc"})
        for m in r["data"]["items"]:
            if int(m["create_time"]) <= before or m["msg_type"] != "text":
                continue
            body = json.loads(m["body"]["content"])["text"]
            if "Access Key" not in body:
                continue
            ak = re.search(r"Access Key:\s*(\S+)", body).group(1)
            sk = re.search(r"Secret Key:\s*(\S+)", body).group(1)
            upd = subprocess.run(["ssh", "dexgem", "python3", "-c", REMOTE_INSTALL],
                                 input=json.dumps({"ak": ak, "sk": sk}),
                                 text=True, capture_output=True)
            print(upd.stdout.strip() or upd.stderr.strip())
            if upd.returncode != 0:
                sys.exit("remote install failed")
            return
    sys.exit("timed out waiting for key delivery")

if __name__ == "__main__":
    main()
