from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import base64
import io
import cv2
import numpy as np
from PIL import Image

app = FastAPI(title="Puzzle Solver API")

# --- CORS tənzimləmələri ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------

class TileData(BaseModel):
    tileId: str
    imageData: str

class PuzzleRequest(BaseModel):
    raw_puzzle_data: List[TileData]
    custom_14_variants: List[List[str]]


def base64_to_cv2(base64_str: str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(img_data)).convert('RGB')
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def crop_fixed_border(img, border_pixels=5):
    """
    Kənar qara çərçivələrin təsirini tamamilə itirmək üçün 
    hər tərəfdən sabit 5 piksel daxili kəsirik.
    """
    h, w, _ = img.shape
    if h <= border_pixels * 2 or w <= border_pixels * 2:
        return img
    return img[border_pixels:h-border_pixels, border_pixels:w-border_pixels]

def get_edge_correlation(edge1, edge2):
    """
    Normalized Cross-Correlation (NCC) vasitəsilə iki kənarın oxşarlığını hesablayır.
    Nəticə -1 ilə 1 arasında olur (1 tam oxşarlıq deməkdir).
    """
    # İki massivi float64 formatına salırıq
    e1 = edge1.astype(np.float64)
    e2 = edge2.astype(np.float64)
    
    # Ortalamadan kənarlaşmanı tapırıq (Z-score normalizasiyası üçün)
    e1_mean = e1 - np.mean(e1, axis=0, keepdims=True)
    e2_mean = e2 - np.mean(e2, axis=0, keepdims=True)
    
    # Kovariasiya və standart meyllərin hasili
    numerator = np.sum(e1_mean * e2_mean)
    denominator = np.sqrt(np.sum(e1_mean ** 2) * np.sum(e2_mean ** 2))
    
    if denominator == 0:
        return 0.0
    
    # Korrelyasiya xalını qaytarırıq
    return numerator / denominator

def check_horizontal_match(piece_left, piece_right):
    left_cropped = crop_fixed_border(piece_left, border_pixels=5)
    right_cropped = crop_fixed_border(piece_right, border_pixels=5)
    
    left_edge = left_cropped[:, -1]
    right_edge = right_cropped[:, 0]
    
    left_edge_inner = left_cropped[:, -2]
    right_edge_inner = right_cropped[:, 1]
    
    # Ən kənar və bir piksel daxildəki zolaqların korrelyasiya ortalaması
    corr1 = get_edge_correlation(left_edge, right_edge)
    corr2 = get_edge_correlation(left_edge_inner, right_edge_inner)
    
    return corr1 * 0.7 + corr2 * 0.3

def check_vertical_match(piece_top, piece_bottom):
    top_cropped = crop_fixed_border(piece_top, border_pixels=5)
    bottom_cropped = crop_fixed_border(piece_bottom, border_pixels=5)
    
    top_edge = top_cropped[-1, :]
    bottom_edge = bottom_cropped[0, :]
    
    top_edge_inner = top_cropped[-2, :]
    bottom_edge_inner = bottom_cropped[1, :]
    
    corr1 = get_edge_correlation(top_edge, bottom_edge)
    corr2 = get_edge_correlation(top_edge_inner, bottom_edge_inner)
    
    return corr1 * 0.7 + corr2 * 0.3


@app.post("/solve")
async def solve_puzzle(payload: PuzzleRequest):
    pieces_map = {}
    
    try:
        for item in payload.raw_puzzle_data:
            pieces_map[item.tileId] = base64_to_cv2(item.imageData)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Görüntülərin base64 emalında xəta: {str(e)}")

    if not pieces_map:
        raise HTTPException(status_code=400, detail="raw_puzzle_data boş ola bilməz.")

    # İndi ən yüksək xalı axtarırıq, ona görə başlanğıcı mənfi sonsuzluq qoyuruq
    best_score = float('-inf')
    best_variant = None

    for variant in payload.custom_14_variants:
        if len(variant) != 9:
            continue
            
        try:
            grid = [
                [pieces_map[variant[0]], pieces_map[variant[1]], pieces_map[variant[2]]],
                [pieces_map[variant[3]], pieces_map[variant[4]], pieces_map[variant[5]]],
                [pieces_map[variant[6]], pieces_map[variant[7]], pieces_map[variant[8]]]
            ]
        except KeyError:
            continue

        current_score = 0

        # Üfüqi qonşuluq oxşarlıqları (Toplanır)
        for row in range(3):
            current_score += check_horizontal_match(grid[row][0], grid[row][1])
            current_score += check_horizontal_match(grid[row][1], grid[row][2])

        # Şaquli qonşuluq oxşarlıqları (Toplanır)
        for col in range(3):
            current_score += check_vertical_match(grid[0][col], grid[1][col])
            current_score += check_vertical_match(grid[1][col], grid[2][col])

        # Burada maksimum oxşarlığı olan variantı seçirik (current_score > best_score)
        if current_score > best_score:
            best_score = current_score
            best_variant = variant

    if best_variant is None:
        raise HTTPException(status_code=422, detail="Uyğun gələn heç bir düzgün variant tapılmadı.")

    return {"best_variant": best_variant}