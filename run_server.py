import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"

import uvicorn

if __name__ == "__main__":
    print("Starting Label Audit Server on http://127.0.0.1:8050 ...")
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8050, log_level="info")
