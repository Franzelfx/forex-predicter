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
    allow_origins=["http://w7cauhlbal7amorf.myfritz.net:4200"],  # Allow the specific frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
    expose_headers=["*"],
)

app.include_router(router, prefix="/v1", tags=["routes"])

inference.init_scheduler()