from src.paper_reader.app import create_app
from src.paper_reader.config import load_config


if __name__ == "__main__":
    runtime_config = load_config()
    create_app(config=runtime_config).run(
        host=runtime_config.host,
        port=runtime_config.port,
        debug=False,
    )
