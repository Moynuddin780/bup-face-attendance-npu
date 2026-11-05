"""Helper run in a subprocess: import an app.py (original or new) and dump its route map
and the HTML of every page for each role, as JSON on stdout.

    python _render_pages.py <app_dir> <requests.json>

requests.json = {"logins": {role: [user, password]}, "pages": {role: [url, ...]}}
"""
import importlib.util
import json
import os
import sys


def main():
    app_dir, req_path = sys.argv[1], sys.argv[2]
    req = json.load(open(req_path))
    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # both apps print banners on import; keep stdout for the JSON
    sys.path.insert(0, app_dir)
    os.chdir(app_dir)
    spec = importlib.util.spec_from_file_location("app_under_test", os.path.join(app_dir, "app.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    app = mod.app
    app.config["TESTING"] = True

    rules = sorted([r.rule, r.endpoint, sorted(r.methods - {"HEAD", "OPTIONS"})] for r in app.url_map.iter_rules())
    out = {"rules": rules, "stream_backend": getattr(mod, "STREAM_BACKEND", None), "pages": {}}
    for role, (user, pw) in req["logins"].items():
        c = app.test_client()
        r = c.post("/", data={"username": user, "password": pw})
        out["pages"][role] = {"__login__": {"status": r.status_code, "location": r.headers.get("Location")}}
        for url in req["pages"][role]:
            r = c.get(url)
            out["pages"][role][url] = {"status": r.status_code, "body": r.get_data(as_text=True)}
    json.dump(out, real_stdout)


if __name__ == "__main__":
    main()
