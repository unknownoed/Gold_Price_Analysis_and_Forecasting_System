from gold_ai.models import Base
from session import engine

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()

# E:\comprehensive_project\venv\gold_ai\db\init_db.py