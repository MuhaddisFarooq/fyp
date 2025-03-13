import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import sys
import cv2
import subprocess
from models.pix2pixHD_model import Pix2PixHDModel
from options.test_options import TestOptions
from util.util import tensor2im, save_image
from diffusers import StableDiffusionUpscalePipeline, DPMSolverMultistepScheduler
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import uvicorn
from fastapi.staticfiles import StaticFiles

# Set the device to use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimize CUDA operations
torch.backends.cudnn.benchmark = True

# [Your existing model definitions: ResidualBlock, EnhancedUNet, etc.]
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class EnhancedUNet(nn.Module):
    def __init__(self):
        super(EnhancedUNet, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down1_residual = ResidualBlock(64)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down2_residual = ResidualBlock(128)
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.down3_residual = ResidualBlock(256)
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.down4_residual = ResidualBlock(512)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.final = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.down1(x)
        x1 = self.down1_residual(x1)
        x2 = self.down2(x1)
        x2 = self.down2_residual(x2)
        x3 = self.down3(x2)
        x3 = self.down3_residual(x3)
        x4 = self.down4(x3)
        x4 = self.down4_residual(x4)
        x = torch.relu(self.up1(x4))
        x = torch.relu(self.up2(x))
        x = torch.relu(self.up3(x))
        x = torch.relu(self.up4(x))
        return self.final(x)

# Load models (unchanged)
diffusion_model = EnhancedUNet().to(device)
diffusion_model_path = r"D:\FINAL_INTEGRATED\FYP Project\new_pix2pix\pix2pix\Pipeline\checkpoints\model_checkpoint.pth"
checkpoint = torch.load(diffusion_model_path)
diffusion_model.load_state_dict(checkpoint['model_state_dict'])
diffusion_model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

def remove_blur_and_noise(image):
    print("Removing blur and noise...")
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
    denoised_image = cv2.GaussianBlur(sharpened_image, (3, 3), 0)
    return denoised_image

def is_blurry(image, threshold=100.0):
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    print(f"Laplacian variance: {laplacian_var}")
    return laplacian_var < threshold

def process_image(image):
    image_cv2 = np.array(image.convert('L'))
    if is_blurry(image_cv2):
        print("Image is blurry. Removing blur...")
        processed_image = remove_blur_and_noise(image_cv2)
    else:
        print("Image is not blurry. Still processing the image (no blur removal).")
        processed_image = image_cv2
    return processed_image

def run_blur_removal_model(image, output_dir):
    processed_image = process_image(image)
    output_image = Image.fromarray(processed_image)
    output_image_path = os.path.join(output_dir, 'blur_removed_output.png')
    output_image.save(output_image_path)
    return output_image_path

sys.argv = ['']
opt = TestOptions().parse(save=False)
opt.name = 'person_face_sketches'
opt.dataroot = r'D:\FINAL_INTEGRATED\FYP Project\new_pix2pix\pix2pix\pix2pixHD\datasets\person_face_sketches'
opt.checkpoints_dir = r'D:\FINAL_INTEGRATED\FYP Project\new_pix2pix\pix2pix\pix2pixHD\checkpoints'
opt.gpu_ids = [0]
opt.no_instance = True
opt.label_nc = 0
opt.loadSize = 256
opt.fineSize = 256
opt.how_many = 1
opt.which_epoch = 'latest'

pix2pix_model = Pix2PixHDModel()
pix2pix_model.initialize(opt)
pix2pix_model.eval()

def preprocess_image_for_pix2pix(image_path, load_size, fine_size):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((load_size, load_size), Image.BICUBIC)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.FloatTensor(img) / 255.0
    img = img.unsqueeze(0).to('cuda')
    return img

def run_pix2pix_model(image_path, output_dir):
    input_image_tensor = preprocess_image_for_pix2pix(image_path, opt.loadSize, opt.fineSize)
    with torch.no_grad():
        generated = pix2pix_model.inference(input_image_tensor, None, None)
    if generated.shape[1] == 1:
        generated = generated.repeat(1, 3, 1, 1)
    output_image_path = os.path.join(output_dir, 'generated_image.png')
    save_output_image(generated, output_image_path)
    return output_image_path

def save_output_image(tensor_image, output_path):
    if len(tensor_image.shape) == 4:
        tensor_image = tensor_image[0]
    image_numpy = tensor2im(tensor_image)
    save_image(image_numpy, output_path)

def load_stable_diffusion_model():
    model_id = r"D:\FINAL_INTEGRATED\FYP Project\new_pix2pix\pix2pix\stablediffusion-main"
    pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(20)
    pipe.disable_attention_slicing()
    return pipe

pipe = load_stable_diffusion_model()

def run_stable_diffusion_model(image_path, output_dir):
    low_res_image = Image.open(image_path).convert("RGB")
    low_res_image = low_res_image.resize((150, 150))
    prompt = "a high-resolution, hyper realistic, extremely detailed 4k image of the person"
    low_res_output_image = pipe(prompt=prompt, image=low_res_image).images[0]
    final_output_image = low_res_output_image.resize((720, 480), Image.NEAREST)
    output_image_path = os.path.join(output_dir, 'upscaled_output.png')
    final_output_image.save(output_image_path)
    return output_image_path

# Initialize FastAPI app
app = FastAPI(title="Sketch-2-Face API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory
output_dir = r"D:\FINAL_INTEGRATED\FYP Project\new_pix2pix\pix2pix\pix2pixHD\output_images"
os.makedirs(output_dir, exist_ok=True)

# Mount static files directory for images
app.mount("/images", StaticFiles(directory=output_dir), name="images")

# Serve the HTML file as the root endpoint
html_file_path = r"D:\FINAL_INTEGRATED\FYP Project\new_pix2pix\pix2pix\test-client.html"  # Update this path as needed
with open(html_file_path, "r", encoding="utf-8") as f:
    html_content = f.read()

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/process/")
async def process_sketch(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        input_path = os.path.join(output_dir, "input.png")
        image.save(input_path)
        print("Running blur-removal model...")
        blur_removed_path = run_blur_removal_model(image, output_dir)
        print("Running pix2pixHD model on blur-removed output...")
        pix2pix_output_path = run_pix2pix_model(blur_removed_path, output_dir)
        input_filename = os.path.basename(input_path)
        blur_removed_filename = os.path.basename(blur_removed_path)
        pix2pix_filename = os.path.basename(pix2pix_output_path)
        return {
            "status": "success",
            "input_image": f"/images/{input_filename}",
            "blur_removed_image": f"/images/{blur_removed_filename}",
            "final_image": f"/images/{pix2pix_filename}"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/output/{filename}")
async def get_image(filename: str):
    image_path = os.path.join(output_dir, filename)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "Image not found"}
    )

@app.get("/test")
async def test_api():
    return {
        "status": "success",
        "models_loaded": {
            "diffusion_model": diffusion_model is not None,
            "pix2pix_model": pix2pix_model is not None,
            "stable_diffusion": pipe is not None
        }
    }

# Optional: Automate ngrok startup
def start_ngrok():
    try:
        # Start ngrok in a subprocess (assumes ngrok executable is in PATH or current directory)
        ngrok_process = subprocess.Popen(
            ["ngrok", "http", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("ngrok tunnel started. Check the ngrok terminal for the public URL.")
        return ngrok_process
    except FileNotFoundError:
        print("Error: ngrok executable not found. Please ensure ngrok is installed and in your PATH.")
        return None

if __name__ == "__main__":
    # Start ngrok (optional)
    ngrok_process = start_ngrok()
    
    # Run the FastAPI app (note: removed any unsupported arguments like debug=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # Cleanup ngrok process when the app stops (optional)
    if ngrok_process:
        ngrok_process.terminate()
