"""Deterministic OpenAI SSE provider for real-Pi tool plumbing smoke tests.

No LLM claim is made: actual Pi, bash, search plugin and HTTP fetch execute;
only tool selection/final prose are scripted to make the regression repeatable.
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import time


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        results = [m for m in body["messages"] if m["role"] == "tool"]
        if results:
            delta = {"content": "Sandbox tool results:\n" + json.dumps(results, ensure_ascii=False)}
            finish = "stop"
        else:
            calls = [
                ("bash", {"command": "pwd; printf 'sandbox scratch\\n' > notes.txt; wc -c materials/source/*"}),
                ("web_search", {"query": "Attention Is All You Need arxiv 1706.03762"}),
                ("web_fetch", {"url": "https://example.com"}),
            ]
            delta = {"role": "assistant", "tool_calls": [
                {"index": i, "id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                for i, (name, args) in enumerate(calls)
            ]}
            finish = "tool_calls"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        base = {"id": "chatcmpl-probe", "object": "chat.completion.chunk", "created": int(time.time()), "model": "sandbox-probe"}
        for choice in [{"index": 0, "delta": delta, "finish_reason": None}, {"index": 0, "delta": {}, "finish_reason": finish}]:
            self.wfile.write(("data: " + json.dumps({**base, "choices": [choice]}) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


HTTPServer(("127.0.0.1", 19867), Handler).serve_forever()
