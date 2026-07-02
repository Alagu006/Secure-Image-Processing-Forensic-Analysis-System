from pydantic import BaseModel, Field
from typing import Optional


class UploadResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None


class ProcessRequest(BaseModel):
    filename: str
    operations: Optional[list[str]] = Field(default_factory=list)


class ProcessResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None


class DownloadResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None


class ScanReportResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    data: Optional[dict] = None
