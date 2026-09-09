"""
JSL-AeroDefect: FastAPI Edge Inspection & Quality Intelligence Service
High-throughput REST & WebSocket endpoints for real-time steel strip defect detection,
periodicity root-cause analysis, grade disposition, and coil digital twin slitting optimization.
"""

import os
import sys
import io
import time
import base64
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional

# Add local backend directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from detector import SteelDefectDetector, DEFECT_CLASSES, generate_synthetic_defect_sample
from periodicity import RollMarkPeriodicityAnalyzer, MILL_ROLL_REGISTRY
from grades import GradeDispositionEngine, GRADE_MATRIX
from coil_twin import CoilDigitalTwin

app = FastAPI(
    title="JSL-AeroDefect Industrial Quality Engine",
    description="AI-Powered Stainless Steel Defect Detection, Root-Cause Diagnostics & Yield Optimizer for Jindal Stainless Limited",
    version="2.0.0"
)

# Enable CORS for industrial SCADA and browser UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
detector = SteelDefectDetector()
periodicity_analyzer = RollMarkPeriodicityAnalyzer()
disposition_engine = GradeDispositionEngine()

SAMPLES_DIR = os.path.join(PROJECT_ROOT, "sample_defects")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# Input Schemas
class RootCauseRequest(BaseModel):
    longitudinal_positions_mm: List[float]

class CoilSimulationRequest(BaseModel):
    coil_id: Optional[str] = "JSL-CR-2026-9842"
    grade: Optional[str] = "304"
    total_length_m: Optional[float] = 1500.0
    strip_width_mm: Optional[float] = 1250.0
    thickness_mm: Optional[float] = 1.5

class CertificateRequest(BaseModel):
    coil_id: str
    heat_no: Optional[str] = "HT-304-8921"
    grade: str
    inspection_standard: Optional[str] = "ASTM A240 / EN 10088-2"
    operator_id: Optional[str] = "QC-INSP-408"
    line_id: Optional[str] = "CRM-LINE-04"
    total_length_m: float
    detections: List[dict]
    disposition: dict

@app.get("/api/health")
def health_check():
    return {
        "status": "ONLINE",
        "engine": "JSL-AeroDefect Pro",
        "device": str(detector.device),
        "cuda_available": hasattr(detector.device, 'type') and detector.device.type == "cuda",
        "timestamp": time.time(),
        "supported_grades": list(GRADE_MATRIX.keys()),
        "defect_classes": [v["name"] for v in DEFECT_CLASSES.values()],
        "torch_available": bool(getattr(detector, 'model', None) is not None)
    }

@app.get("/api/grades")
def get_grades():
    return GRADE_MATRIX

@app.get("/api/samples")
def get_sample_defects():
    samples = []
    if os.path.exists(SAMPLES_DIR):
        for f in sorted(os.listdir(SAMPLES_DIR)):
            if f.endswith(".jpg") or f.endswith(".png"):
                key = f.replace("sample_", "").replace(".jpg", "").upper()
                name = "Clean Prime Strip" if "clean" in f else next((v["name"] for v in DEFECT_CLASSES.values() if v["id"] == key), key)
                samples.append({
                    "filename": f,
                    "defect_code": key,
                    "defect_name": name,
                    "url": f"/sample_defects/{f}"
                })
    return samples

@app.post("/api/inspect")
async def inspect_steel_strip(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
    grade: Optional[str] = Form("304"),
    confidence_threshold: Optional[float] = Form(0.35)
):
    start_time = time.perf_counter()
    image_bytes = None
    
    if file is not None and file.filename != "":
        image_bytes = await file.read()
    elif sample_name:
        sample_path = os.path.join(SAMPLES_DIR, sample_name)
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                image_bytes = f.read()
        else:
            raise HTTPException(status_code=404, detail=f"Sample file {sample_name} not found.")
    else:
        # Fallback to default scratch sample
        default_sample = os.path.join(SAMPLES_DIR, "sample_sc.jpg")
        if os.path.exists(default_sample):
            with open(default_sample, "rb") as f:
                image_bytes = f.read()
        else:
            raise HTTPException(status_code=400, detail="No image provided for inspection.")

    # Decode image to BGR numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

    # Run AI Detection Engine
    det_results = detector.detect(img_bgr, confidence_threshold=confidence_threshold, grade=grade)
    
    # Run Grade Disposition Engine
    disposition = disposition_engine.evaluate(det_results["detections"], grade_key=grade)
    
    # Check if Roll Marks were detected -> Run Periodicity Root-Cause Diagnostics
    roll_marks = [d for d in det_results["detections"] if d["class_id"] == "RM"]
    periodicity_result = None
    if len(roll_marks) >= 2:
        # Extract y-coordinates and scale to simulated millimeters (e.g. 512px = 1000mm)
        px_to_mm = 2.0
        y_positions = [d["center"][1] * px_to_mm for d in roll_marks]
        periodicity_result = periodicity_analyzer.analyze_periodicity(y_positions)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    
    # Generate visual annotated overlay with contours and labels for web rendering
    annotated_img = img_bgr.copy()
    for d in det_results["detections"]:
        # Parse hex color to BGR
        hex_col = d["color"].lstrip("#")
        bgr_col = tuple(int(hex_col[i:i+2], 16) for i in (4, 2, 0))
        
        # Bounding box
        x, y, bw, bh = d["bbox"]
        cv2.rectangle(annotated_img, (x, y), (x + bw, y + bh), bgr_col, 2)
        
        # Draw contour if present
        if d.get("polygon"):
            pts = np.array(d["polygon"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_img, [pts], True, bgr_col, 1, cv2.LINE_AA)
            
        # Text badge with explicit Defect Type and Confidence Score
        label = f"TYPE: {d['class_name']} | CONF: {d['confidence']*100:.1f}% | DSI: {d['dsi']}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0]
        cv2.rectangle(annotated_img, (x, max(0, y - 24)), (x + text_size[0] + 8, y), bgr_col, -1)
        cv2.putText(annotated_img, label, (x + 4, max(15, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    _, buffer = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    return {
        "success": True,
        "inference_latency_ms": elapsed_ms,
        "fps_equivalent": round(1000.0 / max(1.0, elapsed_ms), 1),
        "results": det_results,
        "disposition": disposition,
        "periodicity_analysis": periodicity_result,
        "annotated_image_b64": annotated_b64
    }

@app.post("/api/root-cause")
def analyze_root_cause(request: RootCauseRequest):
    """
    Analyzes spatial roll mark positions to diagnose faulty work roll stand.
    """
    result = periodicity_analyzer.analyze_periodicity(request.longitudinal_positions_mm)
    return result

@app.post("/api/coil/simulate")
def simulate_coil_digital_twin(request: CoilSimulationRequest):
    """
    Generates 2D Coil Digital Twin and computes smart slitting optimization plan.
    """
    twin = CoilDigitalTwin(
        coil_id=request.coil_id,
        grade=request.grade,
        total_length_m=request.total_length_m,
        strip_width_mm=request.strip_width_mm,
        thickness_mm=request.thickness_mm
    )
    defects = twin.generate_simulated_coil_run()
    grade_info = GRADE_MATRIX.get(request.grade, GRADE_MATRIX["304"])
    slitting_plan = twin.compute_smart_slitting_plan(defects, grade_info)
    return slitting_plan

@app.post("/api/certificate")
def generate_quality_certificate(request: CertificateRequest):
    """
    Generates formal ASTM A240 / A480 Coil Inspection Certificate.
    """
    grade_data = GRADE_MATRIX.get(request.grade, GRADE_MATRIX["304"])
    cert_no = f"JSL-CERT-{int(time.time()) % 1000000:06d}"
    
    return {
        "certificate_no": cert_no,
        "issue_date": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "producer": {
            "company": "Jindal Stainless Limited (JSL)",
            "plant": "Jajpur Stainless Steel Complex / Hisar Works",
            "division": "Cold Rolling Mill (CRM) Quality Assurance",
            "iso_certifications": "ISO 9001:2015, IATF 16949:2016, ISO 14001"
        },
        "coil_parameters": {
            "coil_id": request.coil_id,
            "heat_no": request.heat_no,
            "steel_grade": request.grade,
            "grade_full_name": grade_data["name"],
            "surface_finish": grade_data["surface_finish"],
            "inspection_standard": request.inspection_standard,
            "line_id": request.line_id,
            "operator_id": request.operator_id,
            "total_length_m": request.total_length_m
        },
        "inspection_metrics": {
            "total_defects_found": len(request.detections),
            "defect_density_per_100m": round((len(request.detections) / max(1.0, request.total_length_m)) * 100.0, 2),
            "disposition_verdict": request.disposition.get("overall_disposition", "UNKNOWN"),
            "compliance_status": "APPROVED (PRIME)" if "PRIME" in request.disposition.get("overall_disposition", "") else "CONDITIONAL / NON-PRIME",
            "disposition_summary": request.disposition.get("summary", "")
        },
        "critical_flaws_breakdown": [
            {"class_name": d.get("class_name"), "dsi": d.get("dsi"), "severity": d.get("severity")}
            for d in request.detections if d.get("dsi", 0) > 30.0
        ]
    }

# Serve sample defects as static files
app.mount("/sample_defects", StaticFiles(directory=SAMPLES_DIR), name="sample_defects")

# Serve frontend static files if directory exists
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
