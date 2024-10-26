from fastapi import FastAPI

from .routers import example

app = FastAPI(
    title='renameme',
    description='Fill the description',
    version='0.1',
)

app.include_router(example.router)
