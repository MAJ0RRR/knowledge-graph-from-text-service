from fastapi import FastAPI

from .routers import graph

app = FastAPI(
    title='knowledge-graph',
    description='A service that generates a graph of concepts from a given text.',
    version='0.1',
)

app.include_router(graph.router)
