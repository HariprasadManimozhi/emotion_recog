from flask import Flask, request, redirect, url_for, render_template,Response,jsonify
from camera import Emo 
import cv2, os, logging, shutil
from flaskext.mysql import MySQL
import numpy as np
from PIL import Image
import json
import glob
from uuid import uuid4
import FaceDetection.ap
#from dash.dependencies import Input, Output


app = Flask(__name__)
#appd = dash.Dash(__name__, server=app,url_base_pathname='/pathname/')

@app.route("/")
def index():
    return render_template("front.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/app")
def ap():
    return render_template("app.html")

@app.route("/emo")
def emo_front():
    return render_template("emo.html")

@app.route("/link")
def er():
	Emo()
	return render_template("result.html")

@app.route("/emor")
def e1():
    return render_template("result.html")

#@app.route("/dash") 
#def MyDashApp():
#    return appd.generate_table()

def ajax_response(status, msg):
    status_code = "ok" if status else "error"
    return json.dumps(dict(
        status=status_code,
        msg=msg,
    ))
