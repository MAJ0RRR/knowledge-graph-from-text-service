import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

router = APIRouter(
    prefix='',
    responses={404: {'description': 'Not found'}},
)

UPLOAD_DIR = Path('/app/model_connection_example/input_data')
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.get('/', response_class=HTMLResponse)
async def get_upload_form():
    """Serve the HTML form for file upload."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload PDF</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background-color: #f7f7f7;
                margin: 0;
            }
            .upload-container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                text-align: center;
            }
            input[type="file"] {
                margin: 20px 0;
                padding: 10px;
            }
            button {
                padding: 10px 20px;
                font-size: 16px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
            .message {
                margin-top: 20px;
                color: #555;
            }
        </style>
    </head>
    <body>
        <div class="upload-container">
            <h2>Upload a PDF File</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".pdf" required><br>
                <button type="submit">Upload PDF</button>
            </form>
            <div class="message"></div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@router.get(
    '/graph',
    summary='Show graph',
    description='Returns an HTML page being graph representation.',
    response_class=HTMLResponse,
)
async def show_graph():
    """This endpoint serves an HTML file."""
    graph_html_path = Path('/app/model_connection_example/docs/index.html')
    if graph_html_path.is_file():
        graph_html = graph_html_path.read_text(encoding='utf-8')
        download_csv_button = """
        <div style="text-align: center; margin-top: 20px;">
            <button onclick="window.location.href='/download'"
                    style="padding: 10px 20px; font-size: 16px; background-color: #4CAF50; color: white; \
                      border: none; border-radius: 5px; cursor: pointer;">
                Download CSV
            </button>
                        <button onclick="window.location.href='/generate-graph'"
                    style="padding: 10px 20px; font-size: 16px; background-color: #4CAF50; color: white; \
                      border: none; border-radius: 5px; cursor: pointer; margin-top: 20px;">
                Generate Graph
            </button>
        </div>
        <br>
        """
        graph_html = download_csv_button + graph_html
        return HTMLResponse(content=graph_html, status_code=200)
    return HTMLResponse(content='<h1>Graph HTML File Not Found</h1>', status_code=404)


@router.get(
    '/download',
    summary='Download Graph as CSV',
    description='Returns a CSV file for download.',
)
async def download_csv():
    """This endpoint serves a graph CSV file."""
    csv_file_path = Path('/app/model_connection_example/data_output/graph.csv')

    if csv_file_path.is_file():
        return FileResponse(csv_file_path, media_type='text/csv', filename='graph.csv')

    return HTMLResponse(content='<h1>CSV File Not Found</h1>', status_code=404)


@router.post(
    '/upload',
    summary='Upload PDF file',
    description='Uploads a PDF file and saves it to the server.',
)
async def upload_pdf(file: UploadFile):
    """This endpoint allows the user to upload a PDF file."""
    # Check if the uploaded file is a PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF files are allowed')

    # Define the path where the file will be saved
    file_location = UPLOAD_DIR / file.filename

    # Save the uploaded file to the server
    try:
        with file_location.open('wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to save the file: {str(e)}') from e

    # Return the file's location after upload
    return HTMLResponse(content=f'<h1>File {file.filename} uploaded successfully!</h1>', status_code=200)


@router.get(
    '/generate-graph',
    summary='Generate Graph',
    description='Invokes the generate_graph.py script.',
)
async def generate_graph():
    """This endpoint invokes the generate_graph.py script."""
    import subprocess

    try:
        result = subprocess.run(
            ['python', '/app/model_connection_example/generate_graph.py'], check=True, capture_output=True, text=True
        )
        return HTMLResponse(
            content=f'<h1>Graph generated successfully!</h1><pre>{result.stdout}</pre>', status_code=200
        )
    except subprocess.CalledProcessError as e:
        return HTMLResponse(content=f'<h1>Failed to generate graph</h1><pre>{e.stderr}</pre>', status_code=500)
