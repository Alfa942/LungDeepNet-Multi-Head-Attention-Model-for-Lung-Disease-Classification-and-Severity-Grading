"""
Real-Time In-Memory Deep Learning Inference Server for LungDeepNet
Architecture: DenseNet121 Shared Feature Extractor + Dual-Head Spatial-Channel Attention + Severity Head
Listens on http://127.0.0.1:5005
Inference latency: ~130-200ms per radiograph image.
"""

import sys
import os
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from infer import predict_image, get_model_and_classifier

PORT = 5005

class MedicalInferenceHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            model, classifier = get_model_and_classifier()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ready",
                "backbone": "LungDeepNet-DenseNet121",
                "weights": "Pretrained Thoracic Feature Weights (NIH ChestX-ray14 + MIMIC + CheXpert)",
                "classifierLoaded": classifier is not None
            }).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/predict":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)
                img_input = data.get("image")

                if not img_input:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": "No image data provided"}).encode("utf-8"))
                    return

                t0 = time.time()
                res = predict_image(img_input)
                elapsed = time.time() - t0
                res["inference_time_ms"] = round(elapsed * 1000, 1)

                response_bytes = json.dumps(res).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(response_bytes)))
                self.end_headers()
                self.wfile.write(response_bytes)
            except Exception as e:
                err_bytes = json.dumps({"success": False, "error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(err_bytes)))
                self.end_headers()
                self.wfile.write(err_bytes)
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        # Concise clinical engine logging
        sys.stderr.write(f"[LungDeepNet Engine] {format % args}\n")


def run():
    print("[LungDeepNet Engine] Loading DenseNet121 & Multi-Head Attention Classifier into RAM...")
    get_model_and_classifier()
    print(f"[LungDeepNet Engine] Ready! Listening at http://127.0.0.1:{PORT}")
    server = HTTPServer(("127.0.0.1", PORT), MedicalInferenceHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[LungDeepNet Engine] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    run()
