from fastapi import FastAPI
from routes import router
from fastapi.middleware.cors import CORSMiddleware

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
