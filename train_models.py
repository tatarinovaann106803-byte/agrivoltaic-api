#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def clean_data(df, target_col):
    """Очистка данных от NaN и бесконечных значений"""
    # Удаляем строки с NaN в целевой колонке
    df = df.dropna(subset=[target_col])
    
    # Удаляем строки с NaN в feature колонках
    df = df.dropna()
    
    # Проверяем, что все значения конечные
    for col in df.select_dtypes(include=[np.number]).columns:
        df = df[np.isfinite(df[col])]
    
    return df

def train_all_models():
    """Обучение всех моделей"""
    
    os.makedirs("models", exist_ok=True)
    
    # 1. Модель для растениеводства
    if os.path.exists('dataset/crop_farming.csv'):
        df = pd.read_csv('dataset/crop_farming.csv')
        
        # Обработка процентов
        if 'productivity_change' in df.columns:
            df['productivity_change'] = df['productivity_change'].astype(str).str.replace('%', '').str.replace(',', '.').astype(float)
        
        # Очистка данных
        df = clean_data(df, 'productivity_change')
        
        features = ['latitude', 'longitude', 'shade_tolerance', 'optimal_temp', 'water_requirement', 'growing_days']
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) >= 3 and len(df) > 5:
            X = df[available_features].values
            y = df['productivity_change'].values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
            model.fit(X_scaled, y)
            
            joblib.dump(model, 'models/model_crop.pkl')
            joblib.dump(scaler, 'models/scaler_crop.pkl')
            print(f"✅ Модель для растениеводства обучена на {len(df)} строках")
        else:
            print(f"⚠ Недостаточно данных для растениеводства: {len(df)} строк")
    
    # 2. Модель для аквакультуры
    if os.path.exists('dataset/aquaculture.csv'):
        df = pd.read_csv('dataset/aquaculture.csv')
        
        if 'productivity_change' in df.columns:
            df['productivity_change'] = df['productivity_change'].astype(str).str.replace(',', '.').astype(float)
        
        # Очистка данных
        df = clean_data(df, 'productivity_change')
        
        features = ['latitude', 'longitude', 'water_temp', 'oxygen_level', 'stocking_density', 'pond_depth']
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) >= 3 and len(df) > 5:
            X = df[available_features].values
            y = df['productivity_change'].values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
            model.fit(X_scaled, y)
            
            joblib.dump(model, 'models/model_aqua.pkl')
            joblib.dump(scaler, 'models/scaler_aqua.pkl')
            print(f"✅ Модель для аквакультуры обучена на {len(df)} строках")
        else:
            print(f"⚠ Недостаточно данных для аквакультуры: {len(df)} строк")
    
    # 3. Модель для лесного хозяйства
    if os.path.exists('dataset/forestry.csv'):
        df = pd.read_csv('dataset/forestry.csv')
        
        if 'productivity_change' in df.columns:
            df['productivity_change'] = df['productivity_change'].astype(str).str.replace('%', '').astype(float)
        
        # Очистка данных
        df = clean_data(df, 'productivity_change')
        
        features = ['latitude', 'longitude', 'tree_height', 'canopy_density', 'growth_rate', 'wood_density']
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) >= 3 and len(df) > 5:
            X = df[available_features].values
            y = df['productivity_change'].values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
            model.fit(X_scaled, y)
            
            joblib.dump(model, 'models/model_forest.pkl')
            joblib.dump(scaler, 'models/scaler_forest.pkl')
            print(f"✅ Модель для лесного хозяйства обучена на {len(df)} строках")
        else:
            print(f"⚠ Недостаточно данных для лесного хозяйства: {len(df)} строк")
    
    print("\n🎉 Обучение завершено!")

if __name__ == "__main__":
    train_all_models()