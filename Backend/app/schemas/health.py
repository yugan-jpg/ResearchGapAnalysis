from pydantic import BaseModel


class MongoHealth(BaseModel):
    configured: bool
    connected: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    service: str
    database: MongoHealth
