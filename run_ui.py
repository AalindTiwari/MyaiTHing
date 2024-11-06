import json
from functools import wraps
import os
from pathlib import Path
import threading
import uuid
from flask import Flask, request, jsonify, Response, send_file, send_from_directory
from flask_basicauth import BasicAuth
from agent import AgentContext
from initialize import initialize
from python.helpers import files
from python.helpers.files import get_abs_path
from python.helpers.print_style import PrintStyle
from python.helpers.dotenv import load_dotenv
from python.helpers import persist_chat, settings
from werkzeug.utils import secure_filename
import mimetypes
import datetime

# initialize the internal Flask server
app = Flask("app", static_folder=get_abs_path("./webui"), static_url_path="/")
app.config["JSON_SORT_KEYS"] = False  # Disable key sorting in jsonify

lock = threading.Lock()

# Set up basic authentication, name and password from .env variables
app.config["BASIC_AUTH_USERNAME"] = (
    os.environ.get("BASIC_AUTH_USERNAME") or "admin"
)  # default name
app.config["BASIC_AUTH_PASSWORD"] = (
    os.environ.get("BASIC_AUTH_PASSWORD") or "admin"
)  # default pass
basic_auth = BasicAuth(app)

# Configure upload folder``
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'work_dir')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# get context to run Orbian in
def get_context(ctxid: str):
    with lock:
        if not ctxid:
            first = AgentContext.first()
            if first:
                return first
            return AgentContext(config=initialize())
        got = AgentContext.get(ctxid)
        if got:
            return got
        return AgentContext(config=initialize(), id=ctxid)


# Now you can use @requires_auth function decorator to require login on certain pages
def requires_auth(f):
    @wraps(f)
    async def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not (
            auth.username == app.config["BASIC_AUTH_USERNAME"]
            and auth.password == app.config["BASIC_AUTH_PASSWORD"]
        ):
            return Response(
                "Could not verify your access level for that URL.\n"
                "You have to login with proper credentials",
                401,
                {"WWW-Authenticate": 'Basic realm="Login Required"'},
            )
        return await f(*args, **kwargs)

    return decorated


# handle default address, show demo html page from ./test_form.html
@app.route("/", methods=["GET"])
async def test_form():
    return Path(get_abs_path("./webui/index.html")).read_text()


# simple health check, just return OK to see the server is running
@app.route("/ok", methods=["GET", "POST"])
async def health_check():
    return "OK"


# # secret page, requires authentication
# @app.route('/secret', methods=['GET'])
# @requires_auth
# async def secret_page():
#     return Path("./secret_page.html").read_text()


# send message to agent (async UI)
@app.route("/msg", methods=["POST"])
async def handle_message_async():
    return await handle_message(False)


# send message to agent (synchronous API)
@app.route("/msg_sync", methods=["POST"])
async def handle_msg_sync():
    return await handle_message(True)


async def handle_message(sync: bool):
    try:

        # data sent to the server
        input = request.get_json()
        text = input.get("text", "")
        ctxid = input.get("context", "")
        blev = input.get("broadcast", 1)

        # context instance - get or create
        context = get_context(ctxid)

        # print to console and log
        PrintStyle(
            background_color="#6C3483", font_color="white", bold=True, padding=True
        ).print(f"User message:")
        PrintStyle(font_color="white", padding=False).print(f"> {text}")
        context.log.log(type="user", heading="User message", content=text)

        if sync:
            context.communicate(text)
            result = await context.process.result()  # type: ignore
            response = {
                "ok": True,
                "message": result,
                "context": context.id,
            }
        else:

            context.communicate(text)
            response = {
                "ok": True,
                "message": "Message received.",
                "context": context.id,
            }

    except Exception as e:
        response = {
            "ok": False,
            "message": str(e),
        }
        PrintStyle.error(str(e))

    # respond with json
    return jsonify(response)


# pausing/unpausing the agent
@app.route("/pause", methods=["POST"])
async def pause():
    try:

        # data sent to the server
        input = request.get_json()
        paused = input.get("paused", False)
        ctxid = input.get("context", "")

        # context instance - get or create
        context = get_context(ctxid)

        context.paused = paused

        response = {
            "ok": True,
            "message": "Agent paused." if paused else "Agent unpaused.",
            "pause": paused,
        }

    except Exception as e:
        response = {
            "ok": False,
            "message": str(e),
        }
        PrintStyle.error(str(e))

    # respond with json
    return jsonify(response)


# load chats from json
@app.route("/loadChats", methods=["POST"])
async def load_chats():
    try:
        # data sent to the server
        input = request.get_json()
        chats = input.get("chats", [])
        if not chats:
            raise Exception("No chats provided")

        ctxids = persist_chat.load_json_chats(chats)

        response = {
            "ok": True,
            "message": "Chats loaded.",
            "ctxids": ctxids,
        }

    except Exception as e:
        response = {
            "ok": False,
            "message": str(e),
        }
        PrintStyle.error(str(e))

    # respond with json
    return jsonify(response)


# save chats to json
@app.route("/exportChat", methods=["POST"])
async def export_chat():
    try:
        # data sent to the server
        input = request.get_json()
        ctxid = input.get("ctxid", "")
        if not ctxid:
            raise Exception("No context id provided")

        context = get_context(ctxid)
        content = persist_chat.export_json_chat(context)

        response = {
            "ok": True,
            "message": "Chats exported.",
            "ctxid": context.id,
            "content": content,
        }

    except Exception as e:
        response = {
            "ok": False,
            "message": str(e),
        }
        PrintStyle.error(str(e))

    # respond with json
    return jsonify(response)


# restarting with new agent0
@app.route("/reset", methods=["POST"])
async def reset():
    try:

        # data sent to the server
        input = request.get_json()
        ctxid = input.get("context", "")

        # context instance - get or create
        context = get_context(ctxid)
        context.reset()
        persist_chat.save_tmp_chat(context)

        response = {
            "ok": True,
            "message": "Agent restarted.",
        }

    except Exception as e:
        response = {
            "ok": False,
            "message": str(e),
        }
        PrintStyle.error(str(e))

    # respond with json
    return jsonify(response)


# killing context
@app.route("/remove", methods=["POST"])
async def remove():
    try:

        # data sent to the server
        input = request.get_json()
        ctxid = input.get("context", "")

        # context instance - get or create
        AgentContext.remove(ctxid)
        persist_chat.remove_chat(ctxid)

        response = {
            "ok": True,
            "message": "Context removed.",
        }

    except Exception as e:
        response = {
            "ok": False,
            "message": str(e),
        }
        PrintStyle.error(str(e))

    # respond with json
    return jsonify(response)


# Web UI polling
@app.route("/poll", methods=["POST"])
async def poll():
    try:

        # data sent to the server
        input = request.get_json()
        ctxid = input.get("context", None)
        from_no = input.get("log_from", 0)

        # context instance - get or create
        context = get_context(ctxid)

        logs = context.log.output(start=from_no)

        # loop AgentContext._contexts
        ctxs = []
        for ctx in AgentContext._contexts.values():
            ctxs.append(
                {
                    "id": ctx.id,
                    "no": ctx.no,
                    "log_guid": ctx.log.guid,
                    "log_version": len(ctx.log.updates),
                    "log_length": len(ctx.log.logs),
                    "paused": ctx.paused,
                }
            )

        # data from this server
        response = {
            "ok": True,
            "context": context.id,
            "contexts": ctxs,
            "logs": logs,
            "log_guid": context.log.guid,
            "log_version": len(context.log.updates),
            "log_progress": context.log.progress,
            "paused": context.paused,
        }

    except Exception as e:
        response = {
            "ok": False,
            "message": str(e),
        }
        PrintStyle.error(str(e))

    # serialize json with json.dumps to preserve OrderedDict order
    response_json = json.dumps(response)
    return Response(response=response_json, status=200, mimetype="application/json")
    # return jsonify(response)


# get current settings
@app.route("/getSettings", methods=["POST"])
async def get_settings():
    try:

        # data sent to the server
        input = request.get_json()

        set = settings.convert_out(settings.get_settings())

        response = {"ok": True, "settings": set}

    except Exception as e:
        response = {
            "ok": False,
            "message": str(e),
        }
        PrintStyle.error(str(e))

    # respond with json
    return jsonify(response)

# set current settings
@app.route("/setSettings", methods=["POST"])
async def set_settings():
    try:

        # data sent to the server
        input = request.get_json()

        set = settings.convert_in(input)
        set = settings.set_settings(set)

        response = {"ok": True, "settings": set}

    except Exception as e:
        response = {
            "ok": False,
            "message": str(e),
        }
        PrintStyle.error(str(e))

    # respond with json
    return jsonify(response)

# Serve 'Forfiles.html' at '/files'
@app.route('/files')
def serve_files():
    return app.send_static_file('Forfiles.html')

@app.route('/files/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def get_file_info(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return None

    file_info = {
        'name': filename,
        'size': os.path.getsize(file_path),
        'type': mimetypes.guess_type(file_path)[0] or 'Unknown',
        'last_modified': datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
    }
    return file_info

@app.route('/api/files', methods=['GET'])
def list_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    files_info = [get_file_info(file) for file in files]
    return jsonify(files_info)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file.filename is None:
        return jsonify({'error': 'Invalid file name'}), 400
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return jsonify({'message': 'File uploaded successfully'}), 200

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({'message': 'File deleted successfully'}), 200
    else:
        return jsonify({'error': 'File not found'}), 404

def run():
    print("Initializing framework...")

    # load env vars
    load_dotenv()

    # initialize contexts from persisted chats
    persist_chat.load_tmp_chats()

    # Suppress only request logs but keep the startup messages
    from werkzeug.serving import WSGIRequestHandler

    class NoRequestLoggingWSGIRequestHandler(WSGIRequestHandler):
        def log_request(self, code="-", size="-"):
            pass  # Override to suppress request logging

    # run the server on port from .env
    port = int(os.environ.get("WEB_UI_PORT", 0)) or None
    app.run(request_handler=NoRequestLoggingWSGIRequestHandler, host="0.0.0.0", port=port)

# run the internal server
if __name__ == "__main__":
    run()