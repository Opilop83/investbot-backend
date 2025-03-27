from flask import Flask, jsonify, request
import threading
import subprocess
import os

app = Flask(__name__)
bot_process = None

@app.route("/status")
def status():
    global bot_process
    if bot_process and bot_process.poll() is None:
        return jsonify({"status": "running"})
    return jsonify({"status": "stopped"})

@app.route("/start")
def start():
    global bot_process
    if not bot_process or bot_process.poll() is not None:
        bot_process = subprocess.Popen(["python", "trade_bot.py"])
        return jsonify({"status": "started"})
    return jsonify({"status": "already running"})

@app.route("/stop")
def stop():
    global bot_process
    if bot_process and bot_process.poll() is None:
        bot_process.terminate()
        return jsonify({"status": "stopped"})
    return jsonify({"status": "not running"})

@app.route("/log")
def log():
    try:
        with open("bot.log", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return str(e), 500
