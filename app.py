#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import joblib
import os
import urllib.request
import json
from datetime import datetime

app = FastAPI(title="Agrivoltaic Calculator API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WeatherFetcher:
    def get_radiation(self, lat, lon):
        """Получение годовой радиации (кВт·ч/м²/год) из NASA POWER"""
        try:
            url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=ALLSKY_SFC_SW_DWN&community=AG&longitude={lon}&latitude={lat}&start=2023&end=2024&format=JSON"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())
            
            # NASA возвращает среднесуточную радиацию (кВт·ч/м²/день)
            daily_values = []
            for month, value in data['properties']['parameter']['ALLSKY_SFC_SW_DWN'].items():
                if value != -999:
                    daily_values.append(value)
            
            if daily_values:
                # Среднесуточная радиация за год
                avg_daily = np.mean(daily_values)
                # Годовая радиация = среднесуточная × 365
                annual = avg_daily * 365
                return round(annual, 0)
        except Exception as e:
            print(f"Ошибка получения радиации: {e}")
        
        # Fallback: эмпирическая формула
        rad = 1500 - abs(lat) * 8
        return round(max(800, min(2200, rad)), 0)
    
    def get_temperature(self, lat, lon):
        """Получение среднегодовой температуры (°C)"""
        try:
            url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=T2M&community=AG&longitude={lon}&latitude={lat}&start=2023&end=2024&format=JSON"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())
            
            temps = []
            for month, value in data['properties']['parameter']['T2M'].items():
                if value != -999:
                    temps.append(value)
            
            if temps:
                annual = np.mean(temps)
                return round(annual, 1)
        except Exception as e:
            print(f"Ошибка получения температуры: {e}")
        
        # Fallback: эмпирическая формула
        return 15.0

class Calculator:
    def __init__(self):
        self.panel_width = 2.134
        self.panel_height = 1.051
        self.panel_power = 0.445  # 445 Вт
        self.panel_area = self.panel_width * self.panel_height
        self.panel_efficiency = 0.20  # 20% КПД
        
    def solar_energy_pvsyst(self, area_ha, coverage, radiation, lat):
        area_m2 = area_ha * 10000
        panel_area_total = area_m2 * coverage
        num_panels = int(panel_area_total / self.panel_area)
        total_power = num_panels * self.panel_power
        
        # Оптимальный угол наклона
        tilt_optimal = abs(lat) * 0.9 + 5
        tilt_optimal = min(55, max(20, tilt_optimal))
        tilt_factor = np.cos(np.radians(tilt_optimal - abs(lat))) * 0.95 + 0.05
        
        # Коэффициенты потерь
        soiling_loss = 0.97      # загрязнение
        thermal_loss = 0.92      # температурные потери
        inverter_loss = 0.97     # инвертор
        cable_loss = 0.98        # кабели
        mismatch_loss = 0.99     # mismatch
        total_efficiency = soiling_loss * thermal_loss * inverter_loss * cable_loss * mismatch_loss
        
        # Годовая выработка (кВт·ч)
        annual_energy = panel_area_total * radiation * total_efficiency * self.panel_efficiency * tilt_factor
        
        # Удельная выработка
        specific_yield = annual_energy / total_power if total_power > 0 else 0
        
        # Упрощенное помесячное распределение
        monthly_energy = [annual_energy / 12] * 12
        
        return {
            'num_panels': num_panels,
            'total_power': total_power,
            'annual_energy': annual_energy,
            'monthly_energy': monthly_energy,
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
                    print(f"Загружена модель {sector}")
                except Exception as e:
                    print(f"Ошибка загрузки {sector}: {e}")
    
    def predict(self, sector, features):
        if sector not in self.models:
            fallbacks = {'crop': 85, 'aqua': 8, 'forest': 12}
            return fallbacks.get(sector, 50)
        X = np.array([features])
        X_scaled = self.scalers[sector].transform(X)
        return float(self.models[sector].predict(X_scaled)[0])

# Инициализация
weather_fetcher = WeatherFetcher()
calculator = Calculator()
predictor = ModelPredictor()

class CalculationRequest(BaseModel):
    sector: str
    lat: float
    lon: float
    area_ha: float
    coverage: float
    height: float
    energy_price: float
    temp: Optional[float] = None
    
    crop_price: Optional[float] = 30
    base_yield: Optional[float] = 10000
    crop_name: Optional[str] = "Пшеница"
    
    fish_price: Optional[float] = 150
    stocking_density: Optional[float] = 10
    oxygen_level: Optional[float] = 7
    pond_depth: Optional[float] = 3
    fish_name: Optional[str] = "Карп"
    
    wood_price: Optional[float] = 5000
    tree_height: Optional[float] = 15
    canopy_density: Optional[float] = 0.6
    forest_name: Optional[str] = "Сосна"

@app.get("/")
def root():
    return {"service": "Agrivoltaic Calculator API", "version": "2.1", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/radiation")
def get_radiation(lat: float, lon: float):
    radiation = weather_fetcher.get_radiation(lat, lon)
    return {"radiation": radiation, "lat": lat, "lon": lon}

@app.post("/calculate")
def calculate(request: CalculationRequest):
    try:
        radiation = weather_fetcher.get_radiation(request.lat, request.lon)
        auto_temp = weather_fetcher.get_temperature(request.lat, request.lon)
        temp = request.temp if request.temp is not None else auto_temp
        
        energy = calculator.solar_energy_pvsyst(
            request.area_ha, request.coverage, radiation, request.lat
        )
        
        if request.sector == "crop":
            features = [request.lat, request.lon, 0.5, temp, 500.0, 120.0]
            productivity_change = predictor.predict('crop', features)
            product_income = request.base_yield * request.area_ha * (productivity_change / 100) * request.crop_price
            base_income = request.base_yield * request.area_ha * request.crop_price
            economics = calculator.economics(energy, product_income, request.energy_price)
            result = {
                "sector": "crop", "sector_name": "Растениеводство", "culture": request.crop_name,
                "temperature_used": temp, "temperature_source": "NASA POWER" if request.temp is None else "user",
                "productivity": {"value": productivity_change, "unit": "%", "label": "изменение урожайности"},
                "energy": energy, "economics": economics
            }
        
        elif request.sector == "aqua":
            features = [request.lat, request.lon, temp, request.oxygen_level, request.stocking_density, request.pond_depth]
            productivity = predictor.predict('aqua', features)
            product_income = productivity * request.area_ha * request.fish_price * 1000
            base_income = request.stocking_density * request.area_ha * request.fish_price * 1000
            economics = calculator.economics(energy, product_income, request.energy_price, capex_per_kw=70000)
            result = {
                "sector": "aqua", "sector_name": "Аквакультура", "culture": request.fish_name,
                "temperature_used": temp, "temperature_source": "NASA POWER" if request.temp is None else "user",
                "productivity": {"value": productivity, "unit": "т/га", "label": "продуктивность"},
                "energy": energy, "economics": economics
            }
        
        else:
            features = [request.lat, request.lon, request.tree_height, request.canopy_density, 1.0, 0.6]
            productivity = predictor.predict('forest', features)
            product_income = productivity * request.area_ha * request.wood_price
            base_income = 8 * request.area_ha * request.wood_price
            economics = calculator.economics(energy, product_income, request.energy_price)
            result = {
                "sector": "forest", "sector_name": "Лесное хозяйство", "culture": request.forest_name,
                "temperature_used": temp, "temperature_source": "NASA POWER" if request.temp is None else "user",
                "productivity": {"value": productivity, "unit": "м³/га/год", "label": "прирост древесины"},
                "energy": energy, "economics": economics
            }
        
        result["radiation"] = radiation
        result["location"] = {"lat": request.lat, "lon": request.lon}
        result["area_ha"] = request.area_ha
        result["coverage"] = request.coverage
        result["height"] = request.height
        
        return {"success": True, "data": result}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)