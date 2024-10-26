from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(
    prefix='/example',
    tags=['example'],
    responses={404: {'description': 'Not found'}},
)


@router.get('/', summary='Hello World Endpoint', description='Returns a simple greeting message.')
def hello_world():
    """This endpoint returns a JSON object with a greeting message.

    Example response:
    - **Hello**: A static string value "World"

    Useful for testing the API status.
    """
    return {'Hello': 'World'}


# A response model
class HelloResponse(BaseModel):
    """Hello response"""

    Hello: str


@router.put(
    '/',
    summary='Greet by Name',
    description='Returns a greeting message with the provided name.',
    response_model=HelloResponse,
)
def hello_name(name: str):
    """Greet the user by their name.

    This endpoint accepts a query parameter `name` and returns a JSON object with a greeting message.

    - **name**: The name of the person to greet.

    Example response:
    - **Hello**: The name provided in the input.
    """
    return {'Hello': name}
