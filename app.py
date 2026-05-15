#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
import os
import urllib.request
import json
from datetime import datetime

# ========== ИНИЦИАЛИЗАЦИЯ ==========
app = FastAPI(title="Agrivoltaic Calculator API", description="API для расчета агривольтаических систем")

# Настройка CORS для работы с Tilda
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене замените на ваш домен Tilda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== КЛАССЫ ИЗ ВАШЕГО NOTEBOOK ==========

class WeatherFetcher:
    def get_radiation(self, lat, lon):
        try:
            url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=ALLSKY_SFC_SW_DWN&community=AG&longitude={lon}&latitude={lat}&start=2023&end=2024&format=JSON"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            rads = [v for v in data['properties']['parameter']['ALLSKY_SFC_SW_DWN'].values() if v != -999]
            if rads:
                annual = np.mean(rads) * 365
                return round(annual, 0)
        except:
            pass
        rad = 1500 - abs(lat) * 8
        return round(max(800, min(2200, rad)), 0)

class Calculator:
    def __init__(self):
        self.panel_width = 2.134
        self.panel_height = 1.051
        self.panel_power = 0.445
        self.panel_area = self.panel_width * self.panel_height
        
    def monthly_radiation_pvsyst(self, lat, radiation_annual):
        months = np.arange(1, 13)
        declination = -23.45 * np.cos(2 * np.pi * (months - 1) / 12)
        sun_alt = 90 - np.abs(lat - declination)
        sun_alt = np.clip(sun_alt, 10, 90)
        monthly_factor = np.sin(np.radians(sun_alt)) / np.sin(np.radians(90 - np.abs(lat)))
        monthly_factor = monthly_factor / monthly_factor.sum() * 12
        return radiation_annual / 12 * monthly_factor
    
    def solar_energy_pvsyst(self, area_ha, coverage, radiation, lat):
        area_m2 = area_ha * 10000
        panel_area_total = area_m2 * coverage
        num_panels = int(panel_area_total / self.panel_area)
        total_power = num_panels * self.panel_power
        
        tilt_optimal = abs(lat) * 0.9 + 5
        tilt_optimal = min(55, max(20, tilt_optimal))
        
        tilt_factor = np.cos(np.radians(tilt_optimal - abs(lat))) * 0.95 + 0.05
        
        soiling_loss = 0.97
        thermal_loss = 0.92
        inverter_loss = 0.97
        cable_loss = 0.98
        mismatch_loss = 0.99
        
        total_efficiency = soiling_loss * thermal_loss * inverter_loss * cable_loss * mismatch_loss
        
        annual_energy = panel_area_total * radiation * total_efficiency * tilt_factor * 0.25
        
        monthly_rad = self.monthly_radiation_pvsyst(lat, radiation)
        monthly_energy = panel_area_total * monthly_rad * total_efficiency * tilt_factor * 0.25
        
        specific_yield = annual_energy / total_power if total_power > 0 else 0
        
        return {
            'num_panels': num_panels,
            'total_power': total_power,
            'annual_energy': annual_energy,
            'monthly_energy': monthly_energy.tolist(),
            'tilt_angle': tilt_optimal,
            'specific_yield': specific_yield,
            'total_efficiency': total_efficiency
        }
    
    def economics(self, energy, product_income, energy_price, capex_per_kw=60000):
        energy_income = energy['annual_energy'] * energy_price
        total_income = energy_income + product_income
        capex = energy['total_power'] * capex_per_kw
        opex = capex * 0.015
        net_income = total_income - opex
        roi = capex / net_income if net_income > 0 else 999
        return {
            'energy_income': energy_income,
            'total_income': total_income,
            'capex': capex,
            'net_income': net_income,
            'roi_years': roi
        }

# ========== ЗАГРУЗКА МОДЕЛЕЙ ==========

class ModelPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.load_models()
    
    def load_models(self):
        model_files = {
            'crop': ('models/model_crop.pkl', 'models/scaler_crop.pkl'),
            'aqua': ('models/model_aqua.pkl', 'models/scaler_aqua.pkl'),
            'forest': ('models/model_forest.pkl', 'models/scaler_forest.pkl')
        }
        
        for sector, (model_path, scaler_path) in model_files.items():
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[sector] = joblib.load(model_path)
                    self.scalers[sector] = joblib.load(scaler_path)
                    print(f"✅ Загружена модель {sector}")
                except Exception as e:
                    print(f"⚠ Ошибка загрузки {sector}: {e}")
    
    def predict(self, sector, features):
        if sector not in self.models:
            # Fallback предсказания
            fallbacks = {'crop': 85, 'aqua': 8, 'forest': 12}
            return fallbacks.get(sector, 50) + np.random.normal(0, 5)
        
        X = np.array([features])
        X_scaled = self.scalers[sector].transform(X)
        return float(self.models[sector].predict(X_scaled)[0])

# ========== PYDANTIC МОДЕЛИ ДЛЯ ЗАПРОСОВ ==========

class CalculationRequest(BaseModel):
    sector: str  # crop, aqua, forest
    lat: float
    lon: float
    area_ha: float
    coverage: float
    height: float
    energy_price: float
    temp: float = 15.0
    
    # Растениеводство
    crop_price: Optional[float] = 30
    base_yield: Optional[float] = 10000
    crop_name: Optional[str] = "Пшеница"
    
    # Аквакультура
    fish_price: Optional[float] = 150
    stocking_density: Optional[float] = 10
    oxygen_level: Optional[float] = 7
    pond_depth: Optional[float] = 3
    fish_name: Optional[str] = "Карп"
    
    # Лесное хозяйство
    wood_price: Optional[float] = 5000
    tree_height: Optional[float] = 15
    canopy_density: Optional[float] = 0.6
    forest_name: Optional[str] = "Сосна"

# ========== API ENDPOINTS ==========

weather_fetcher = WeatherFetcher()
calculator = Calculator()
predictor = ModelPredictor()

@app.get("/")
def root():
    return {
        "service": "Agrivoltaic Calculator API",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "/calculate": "POST - Расчет агривольтаической системы",
            "/radiation": "GET - Получение радиации по координатам",
            "/health": "GET - Проверка состояния"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/radiation")
def get_radiation(lat: float, lon: float):
    """Получение солнечной радиации по координатам"""
    radiation = weather_fetcher.get_radiation(lat, lon)
    return {"radiation": radiation, "lat": lat, "lon": lon}

@app.post("/calculate")
def calculate(request: CalculationRequest):
    """Основной расчет агривольтаической системы"""
    
    try:
        # Получаем радиацию
        radiation = weather_fetcher.get_radiation(request.lat, request.lon)
        
        # Расчет энергии
        energy = calculator.solar_energy_pvsyst(
            request.area_ha, 
            request.coverage, 
            radiation, 
            request.lat
        )
        
        # Расчет в зависимости от сектора
        if request.sector == "crop":
            # Растениеводство
            features = [request.lat, request.lon, 0.4, request.temp, 500]
            productivity_change = predictor.predict('crop', features)
            
            product_income = request.base_yield * request.area_ha * (productivity_change / 100) * request.crop_price
            base_income = request.base_yield * request.area_ha * request.crop_price
            economics = calculator.economics(energy, product_income, request.energy_price)
            
            result = {
                "sector": "crop",
                "sector_name": "Растениеводство",
                "culture": request.crop_name,
                "productivity": {
                    "value": productivity_change,
                    "unit": "%",
                    "label": "изменение урожайности"
                },
                "energy": {
                    "annual_kwh": energy['annual_energy'],
                    "power_kw": energy['total_power'],
                    "specific_yield": energy['specific_yield'],
                    "panels": energy['num_panels'],
                    "tilt_angle": energy['tilt_angle'],
                    "monthly_energy": energy['monthly_energy']
                },
                "economics": {
                    "capex_rub": economics['capex'],
                    "net_income_rub": economics['net_income'],
                    "roi_years": economics['roi_years'],
                    "energy_income_rub": economics['energy_income'],
                    "product_income_rub": product_income,
                    "base_income_rub": base_income,
                    "total_income_rub": economics['total_income']
                }
            }
            
        elif request.sector == "aqua":
            # Аквакультура
            features = [request.lat, request.lon, request.temp, request.oxygen_level, request.stocking_density, request.pond_depth]
            productivity = predictor.predict('aqua', features)
            
            product_income = productivity * request.area_ha * request.fish_price * 1000
            base_income = request.stocking_density * request.area_ha * request.fish_price * 1000
            economics = calculator.economics(energy, product_income, request.energy_price, capex_per_kw=70000)
            
            result = {
                "sector": "aqua",
                "sector_name": "Аквакультура",
                "culture": request.fish_name,
                "productivity": {
                    "value": productivity,
                    "unit": "т/га",
                    "label": "продуктивность"
                },
                "energy": {
                    "annual_kwh": energy['annual_energy'],
                    "power_kw": energy['total_power'],
                    "specific_yield": energy['specific_yield'],
                    "panels": energy['num_panels'],
                    "tilt_angle": energy['tilt_angle'],
                    "monthly_energy": energy['monthly_energy']
                },
                "economics": {
                    "capex_rub": economics['capex'],
                    "net_income_rub": economics['net_income'],
                    "roi_years": economics['roi_years'],
                    "energy_income_rub": economics['energy_income'],
                    "product_income_rub": product_income,
                    "base_income_rub": base_income,
                    "total_income_rub": economics['total_income']
                }
            }
            
        else:  # forest
            # Лесное хозяйство
            features = [request.lat, request.lon, request.tree_height, request.canopy_density, 0.8]
            productivity = predictor.predict('forest', features)
            
            product_income = productivity * request.area_ha * request.wood_price
            base_income = 8 * request.area_ha * request.wood_price
            economics = calculator.economics(energy, product_income, request.energy_price)
            
            result = {
                "sector": "forest",
                "sector_name": "Лесное хозяйство",
                "culture": request.forest_name,
                "productivity": {
                    "value": productivity,
                    "unit": "м³/га/год",
                    "label": "прирост древесины"
                },
                "energy": {
                    "annual_kwh": energy['annual_energy'],
                    "power_kw": energy['total_power'],
                    "specific_yield": energy['specific_yield'],
                    "panels": energy['num_panels'],
                    "tilt_angle": energy['tilt_angle'],
                    "monthly_energy": energy['monthly_energy']
                },
                "economics": {
                    "capex_rub": economics['capex'],
                    "net_income_rub": economics['net_income'],
                    "roi_years": economics['roi_years'],
                    "energy_income_rub": economics['energy_income'],
                    "product_income_rub": product_income,
                    "base_income_rub": base_income,
                    "total_income_rub": economics['total_income']
                }
            }
        
        # Добавляем общую информацию
        result["radiation"] = radiation
        result["location"] = {"lat": request.lat, "lon": request.lon}
        result["area_ha"] = request.area_ha
        result["coverage"] = request.coverage
        result["height"] = request.height
        
        return {"success": True, "data": result}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# ========== ЗАПУСК ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)