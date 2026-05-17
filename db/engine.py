from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session

engine = create_engine("sqlite:///warren.db", echo=False)


class Base(DeclarativeBase):
    pass


def get_session() -> Session:
    return Session(engine)


def init_db() -> None:
    Base.metadata.create_all(engine)
