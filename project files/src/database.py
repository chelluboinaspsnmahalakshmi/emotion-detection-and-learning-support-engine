import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import bcrypt

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    UserID = Column(Integer, primary_key=True, autoincrement=True)
    Name = Column(String, nullable=False)
    Email = Column(String, unique=True, nullable=False)
    Password = Column(String, nullable=False)
    Role = Column(String, default='student')
    Login_Count = Column(Integer, default=0)
    Created_At = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship back to Emotion_Records
    emotion_records = relationship("Emotion_Record", back_populates="user")

    def set_password(self, password: str):
        self.Password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password: str) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), self.Password.encode('utf-8'))

class Emotion_Record(Base):
    __tablename__ = 'emotion_records'

    RecordID = Column(Integer, primary_key=True, autoincrement=True)
    UserID = Column(Integer, ForeignKey('users.UserID'), nullable=False)
    Email = Column(String, nullable=False)
    Field = Column(String, nullable=True)
    Input_Text = Column(Text, nullable=False)
    Predicted_Emotion = Column(String, nullable=False)
    Secondary_Emotion = Column(String, nullable=True)
    Confidence_Score = Column(Float, nullable=True)
    Model_Used = Column(String, nullable=True)
    AI_Response = Column(Text, nullable=True)
    Response_Type = Column(String, nullable=True)
    Emotion_Scores = Column(Text, nullable=True)  # JSON string
    Timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    CSV_Logged = Column(Boolean, default=False)

    user = relationship("User", back_populates="emotion_records")

# Setup SQLite Database
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'app.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(f'sqlite:///{DB_PATH}', echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
