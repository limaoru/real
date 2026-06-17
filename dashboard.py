"""向后兼容。推荐: python -m retail dashboard"""
from retail.apps.dashboard import serve

if __name__ == "__main__":
    serve()
