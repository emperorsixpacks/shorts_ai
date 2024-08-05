from flask import Flask

from shortsai.server.routes import router
from shortsai.settings import ServerSettings

server_settings = ServerSettings()

app = Flask(import_name="shorts ai server")

app.register_blueprint(router)

if __name__ == "__main__":
    app.run(
        host=server_settings.server_host,
        port=server_settings.server_port,
        debug=server_settings.debug,
    )
    
