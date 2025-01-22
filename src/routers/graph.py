import secrets
import shutil
import subprocess
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, Request, Query
from fastapi.responses import FileResponse, HTMLResponse

#########################################
# Router Setup
#########################################
router = APIRouter(
    prefix='',
    responses={404: {'description': 'Not found'}},
)

#########################################
# Helper HTML Components
#########################################
back_to_upload_button_template = """
    <div style="text-align: center; margin-top: 20px;">
        <button onclick="window.location.href='/?session_id={sess_id}'"
        style="padding: 10px 20px; font-size: 16px; background-color: #4CAF50; color: white; \
            border: none; border-radius: 5px; cursor: pointer;">
        Back to Upload Page
        </button>
    </div>
    <br>
"""

#########################################
# Main Pages & Endpoints
#########################################

@router.get('/', response_class=HTMLResponse)
async def get_upload_form(session_id: str = Query(None)):
    """
    Serve the HTML form for file upload.
    We generate or reuse a session_id (if provided).
    """
    # If user didn't provide a session_id in the query, generate a new one
    if not session_id:
        session_id = secrets.token_urlsafe(16)

    back_to_upload_button = back_to_upload_button_template.format(sess_id=session_id)

    # The form action includes ?session_id=...
    # The "View Graph" button includes ?session_id=...
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <title>Upload PDF</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background-color: #f7f7f7;
                margin: 0;
            }}
            .upload-container {{
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                text-align: center;
            }}
            input[type="file"] {{
                margin: 20px 0;
                padding: 10px;
            }}
            button {{
                padding: 10px 20px;
                font-size: 16px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #45a049;
            }}
            .message {{
                margin-top: 20px;
                color: #555;
            }}
        </style>
    </head>
    <body>
        <div class="upload-container">
            <h2>Upload a PDF File</h2>
            <form action="/upload?session_id={session_id}" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".pdf" required><br>
                <button type="submit">Upload PDF</button>
            </form>
            <div class="message"></div>
            <button onclick="window.location.href='/graph?session_id={session_id}'">View Graph</button>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@router.post(
    '/upload',
    summary='Upload PDF file',
    description='Uploads a PDF file and saves it to the server.'
)
async def upload_pdf(
    file: UploadFile,
    session_id: str = Query(...),
):
    """
    Endpoint for uploading a PDF file to a session-specific directory.
    session_id is taken from the query parameter.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF files are allowed')

    # Create a session-specific upload directory
    upload_dir = Path(f'/app/model_connection_example/input_data/{session_id}')
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Define the path where the file will be saved
    file_location = upload_dir / file.filename

    # Save the uploaded file
    try:
        with file_location.open('wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to save the file: {str(e)}') from e

    # Return success message
    back_to_upload_button = back_to_upload_button_template.format(sess_id=session_id)
    return HTMLResponse(
        content=f'<h1>File {file.filename} uploaded successfully!</h1>{back_to_upload_button}',
        status_code=200
    )


@router.get(
    '/generate-graph',
    summary='Generate Graph',
    description='Invokes the generate_graph.py script for this session.'
)
async def generate_graph(session_id: str = Query(...)):
    """
    This endpoint calls a script to generate the graph for the specific session.
    We pass session_id as an argument to generate_graph.py, which reads/writes the
    session-specific directories.
    """
    back_to_upload_button = back_to_upload_button_template.format(sess_id=session_id)
    try:
        result = subprocess.run(
            [
                'python',
                '/app/model_connection_example/generate_graph.py',
                session_id  # pass the session_id as an argument
            ],
            check=True,
            capture_output=True,
            text=True
        )

        return HTMLResponse(
            content=(
                f'<h1>Graph generated successfully!</h1>'
                f'<pre>{result.stdout}</pre>'
                f'{back_to_upload_button}'
            ),
            status_code=200
        )
    except subprocess.CalledProcessError as e:
        return HTMLResponse(
            content=(
                f'<h1>Failed to generate graph</h1>'
                f'<pre>{e.stderr}</pre>'
                f'{back_to_upload_button}'
            ),
            status_code=500
        )


@router.get(
    '/graph',
    summary='Show graph',
    description='Returns an HTML page with the graph representation.',
    response_class=HTMLResponse,
)
async def show_graph(session_id: str = Query(...)):
    """
    This endpoint serves the session-specific HTML file located at:
    /app/model_connection_example/docs/{session_id}/index.html
    """
    graph_html_path = Path(f'/app/model_connection_example/docs/{session_id}/index.html')
    back_to_upload_button = back_to_upload_button_template.format(sess_id=session_id)

    if graph_html_path.is_file():
        graph_html = graph_html_path.read_text(encoding='utf-8')
        download_and_generate_buttons = f"""
        <div style="text-align: center; margin-top: 20px;">
            <button onclick="window.location.href='/download?session_id={session_id}'"
                    style="padding: 10px 20px; font-size: 16px; background-color: #4CAF50; color: white; \
                          border: none; border-radius: 5px; cursor: pointer;">
                Download CSV
            </button>
            <button onclick="window.location.href='/generate-graph?session_id={session_id}'"
                    style="padding: 10px 20px; font-size: 16px; background-color: #4CAF50; color: white; \
                          border: none; border-radius: 5px; cursor: pointer; margin-top: 20px;">
                Generate Graph
            </button>
        </div>
        <br>
        {back_to_upload_button}
        """
        # Prepend the buttons to the HTML
        return HTMLResponse(content=download_and_generate_buttons + graph_html, status_code=200)

    content = back_to_upload_button + '<h1>Graph HTML File Not Found</h1>'
    content += f"""
        <div style="text-align: center; margin-top: 20px;">
            <button onclick="window.location.href='/generate-graph?session_id={session_id}'"
                    style="padding: 10px 20px; font-size: 16px; background-color: #4CAF50; color: white; \
                          border: none; border-radius: 5px; cursor: pointer; margin-top: 20px;">
                Generate Graph
            </button>
        </div>
    """
    return HTMLResponse(content=content, status_code=404)


@router.get(
    '/download',
    summary='Download Graph as CSV',
    description='Returns a CSV file for download.',
)
async def download_csv(session_id: str = Query(...)):
    """
    Endpoint for downloading the session-specific CSV file located at:
    /app/model_connection_example/data_output/{session_id}/graph.csv
    """
    csv_file_path = Path(f'/app/model_connection_example/data_output/{session_id}/graph.csv')

    if csv_file_path.is_file():
        return FileResponse(csv_file_path, media_type='text/csv', filename='graph.csv')
    return HTMLResponse(content='<h1>CSV File Not Found</h1>', status_code=404)
