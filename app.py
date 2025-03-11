from typing import Union
from fastapi import FastAPI
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
import base64
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


# @app.get("/")
# def main():
#     return "hello"

@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return templates.TemplateResponse(
        request=request, name="item.html", context={"id": id}
    )

@app.get("/upload")
async def upload_file(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/display")
async def display_file(request: Request, file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open("contents/uploaded_" + file.filename, "wb") as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    base64_encoded_image = base64.b64encode(contents).decode("utf-8")

    return templates.TemplateResponse("display.html", {"request": request,  "myImage": base64_encoded_image})


def extract_text(file: UploadFile):
    if file.filename.endswith(".txt"):
        return file.file.read().decode("utf-8")
    elif file.filename.endswith(".pdf"):
        reader = PdfReader(file.file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.filename.endswith(".xml"):
        tree = ET.parse(file.file)
        return ET.tostring(tree.getroot(), encoding='utf-8').decode()
    elif file.filename.endswith(".json"):
        return json.dumps(json.load(file.file), indent=4)
    elif file.filename.endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(file.file)
        return pytesseract.image_to_string(image)
    else:
        return "Unsupported file format"

def translate_text(text: str):
    return f"[Translated] {text}"  # Placeholder, integrate a real translation model

def summarize_text(text: str):
    return text[:200] + "..." if len(text) > 200 else text  # Placeholder summary

def extract_keywords(text: str):
    words = text.split()
    return list(set(words[:10]))  # Placeholder keyword extraction

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_file(
    file: UploadFile = File(...),
    process_type: str = Form(...),
    output_format: str = Form("json"),
    request: Request = None
):
    text = extract_text(file)
    if text == "Unsupported file format":
        return JSONResponse({"error": "Unsupported file format"}, status_code=400)

    result = ""
    if process_type == "translation":
        result = translate_text(text)
    elif process_type == "summarization":
        result = summarize_text(text)
    elif process_type == "keywords":
        result = extract_keywords(text)
    else:
        return JSONResponse({"error": "Invalid process type"}, status_code=400)

    if output_format == "text":
        return PlainTextResponse(result if isinstance(result, str) else " ".join(result))
    elif output_format == "xml":
        root = ET.Element("Response")
        child = ET.SubElement(root, process_type)
        child.text = result if isinstance(result, str) else ", ".join(result)
        return Response(ET.tostring(root), media_type="application/xml")
    elif output_format == "html" and request:
        return templates.TemplateResponse("result.html", {"request": request, "result": result})
    else:  # Default JSON
        return JSONResponse({"result": result})