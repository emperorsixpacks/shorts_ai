from flask import Flask

TEMPLATES_LOCATION = ""
STATIC_FILES_LOCATION = ""

app =  Flask(import_name="shorts ai server")

@app.get("/")
def index_page():
    return "hello"