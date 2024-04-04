import os
import json
from pytz import utc
from fastapi import FastAPI
from src.routes import router
from src.composer import Composer
from src.logger import logger as loguru
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import src.inference as inference
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, replace with your domain(s)
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
    allow_credentials=True,  # Allow sending cookies and credentials
    expose_headers=["*"],  # Expose all response headers
)
app.include_router(router, prefix="/v1", tags=["routes"])

inference.init_scheduler()