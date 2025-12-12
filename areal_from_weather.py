import requests
import json
import csv
import os
from datetime import datetime
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from shapely.geometry import Point, box
from shapely.ops import unary_union
from rasterio.mask import geometry_mask
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import KFold

REFERENCE_RASTER_PATH = "./weather/wc2.1_10m_bio_1.tif"
PATH_TO_WORLDCLIM = "./weather" 
EUROPE_BBOX = (-25.0, 34.0, 45.0, 72.0)



def jq_filter(observations):
    filtered_data = []
    for result in observations:
        row = [
            result.get('observed_on_details', {}).get('date', ''),
            result.get('time_observed_at', ''),
            result.get('created_time_zone', '') or result.get('created__time_zone', ''),
            result.get('location', ''),
        ]
        filtered_data.append(row)
    return filtered_data


def fetch_month_data(year, month, taxon_id, place_id):
    if month == 12:
        next_year = year + 1
        next_month = 1
    else:
        next_year = year
        next_month = month + 1
    
    d1 = f"{year}-{month:02d}-01"
    d2 = f"{next_year}-{next_month:02d}-01"
    
    month_names = [
        "январь", "февраль", "март", "апрель", "май", "июнь",
        "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"
    ]
    period_name = f"{month_names[month-1]} {year} года"
    
    print(f"Загрузка данных за {period_name}...")
    
    all_observations_data = []
    page = 1
    has_more_data = True
    success = True
    
    base_url = f"https://api.inaturalist.org/v1/observations?d1={d1}&d2={d2}&taxon_id={taxon_id}&place_id={place_id}&per_page=200&order=desc&order_by=created_at&quality_grade=research"
    
    while has_more_data:
        url = f"{base_url}&page={page}"
        try:
            response = requests.get(
                url,
                headers={'Accept': 'application/json'},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            observations = data.get('results', [])
            
            has_more_data = len(observations) > 0 and len(observations) == 200
            
            if observations:
                all_observations_data.extend(observations)
                page += 1
                
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при загрузке страницы {page} за {period_name}: {e}")
            success = False
            has_more_data = False
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON на странице {page} за {period_name}: {e}")
            success = False
            has_more_data = False
    
    filtered_data = jq_filter(all_observations_data) if all_observations_data else []
    return filtered_data, success

def get_data(taxon_id):

    place_id = "97391"   
    
    headers = ["data", "time", "time_zone", "place"]
    current_year = datetime.today().year
    current_month = datetime.today().month
    
    all_data = []
    month_names = [
                    "январь", "февраль", "март", "апрель", "май", "июнь",
                    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"
                ]
    for year in range(2014, current_year + 1):
        if year < current_year:
            months_range = range(1, 13)
        else:
            months_range = range(1, current_month + 1)
        
        for month in months_range:
            month_data, success = fetch_month_data(year, month, taxon_id, place_id)
            
            if success and month_data:
                print(f"Загружено {len(month_data)} записей за {month_names[month-1]} {year} года")
                
                all_data.extend(month_data)
            elif not success:
                print(f"Не удалось загрузить данные за {month_names[month-1]} {year} года")
    sorted_data = []
    if all_data:
        sorted_data = sorted(all_data, key=lambda x: x[0] if x[0] else '')
        with open(taxon_id+'_all.csv', 'w', newline='', encoding='utf-8') as all_file:
            writer = csv.writer(all_file)
            writer.writerow(headers)
            writer.writerows(sorted_data)
        print(f"Все данные сохранены в "+taxon_id+"_all.csv)")
    return sorted_data


def parse_coords(coord_str):
    clean_str = coord_str.replace('"', '').strip()
    try:
        lat, lon = map(float, clean_str.split(','))
        return lat, lon
    except ValueError:
        return None, None
    
def clean_data():
    taxon_id = input("введите ID вида:")
    if os.path.exists(taxon_id+'_all.csv'):
        df = pd.read_csv(taxon_id+'_all.csv')
    else:
       get_data(taxon_id)
       df = pd.read_csv(taxon_id+'_all.csv')
    df[['lat', 'lon']] = df["place"].apply(lambda x: pd.Series(parse_coords(x)))

    df = df.dropna(subset=['lat', 'lon'])

    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    europe_poly = box(*EUROPE_BBOX)

    gdf_europe = gdf[gdf.geometry.within(europe_poly)].copy()
    print(f"Записей в Европе: {len(gdf_europe)}")

    rows, cols = [], []

    with rasterio.open(REFERENCE_RASTER_PATH) as src:
        for geometry in gdf_europe.geometry:
            x, y = geometry.x, geometry.y
            row, col = src.index(x, y)
            rows.append(row)
            cols.append(col)
        
        gdf_europe['grid_id'] = [f"{r}_{c}" for r, c in zip(rows, cols)]
        gdf_clean = gdf_europe.drop_duplicates(subset=['grid_id'], keep='first')
        
        print(f"Записей после удаления дублей: {len(gdf_clean)}")
        return gdf_clean

def data_absens(gdf_clean):
    num_presence = len(gdf_clean)
    num_absence_needed = num_presence * 2
    grid_ids_presence = set(gdf_clean['grid_id'].values)
    absence_points = []

    with rasterio.open(REFERENCE_RASTER_PATH) as src:
        raster_data = src.read(1)
        nodata_val = src.nodata
        transform = src.transform
        width = src.width
        height = src.height
        
        west, south, east, north = EUROPE_BBOX
        row_min, col_min = src.index(west, north)
        row_max, col_max = src.index(east, south)
        
        row_min = max(0, row_min); col_min = max(0, col_min)
        row_max = min(height, row_max); col_max = min(width, col_max)
        
        count = 0
        cc = 0
        while (count < num_absence_needed) or(cc==num_absence_needed*10):
            rand_r = np.random.randint(row_min, row_max)
            rand_c = np.random.randint(col_min, col_max)
            
            val = raster_data[rand_r, rand_c]
            if val != nodata_val and not np.isnan(val):
                lon, lat = rasterio.transform.xy(transform, rand_r, rand_c, offset='center')
                grid_id = f"{rand_r}_{rand_c}"
                if not( grid_id in grid_ids_presence ):
                    absence_points.append({'lat': lat, 'lon': lon, 'species_present': 0})
                    count += 1
            cc += 1

    return  pd.DataFrame(absence_points)


def connect_data():
    df_data = clean_data()
    df_presence_final = pd.DataFrame(df_data.drop(columns=['geometry', 'grid_id']))
    df_presence_final['species_present'] = 1

    final_dataset = pd.concat([df_presence_final,data_absens(df_data)], ignore_index=True)
    final_dataset = final_dataset.sample(frac=1).reset_index(drop=True)

    final_dataset.to_csv("training_data_europe.csv", index=False)
    return final_dataset

def weather_data():
    df = connect_data()

    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    bio_vars = [f"{i}" for i in range(1, 20)] 
    coords = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]

    for var in bio_vars:
        filename = f"wc2.1_10m_bio_{var}.tif" 
        filepath = os.path.join(PATH_TO_WORLDCLIM, filename)
        
        if  os.path.exists(filepath):
            try:
                with rasterio.open(filepath) as src:
                    values = [x[0] for x in src.sample(coords)]
                    gdf[f'BIO{var}'] = values
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")
        else:
            print(f"Файл {filename} не найден, пропускаем.")

    final_df = pd.DataFrame(gdf.drop(columns=['geometry'] ))
    bio_cols = [f"BIO{i}" for i in range(1, 20)]
    final_df = final_df.dropna(subset=bio_cols)
    return final_df


def model():
    df = weather_data()
    bio_cols = [f"BIO{i}" for i in range(1, 20)]
    X = df[bio_cols]
    y = df['species_present']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Обучающая выборка: {X_train.shape}, Тестовая: {X_test.shape}")

    rf_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=12, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced' 
    )

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    print("\n--- Результаты валидации ---")
    print(classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {roc_auc:.4f}")
    # 5. График важности признаков
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.title("Топ факторов, влияющих на распространение вида")
    sns.barplot(x=importances[indices][:10], y=[bio_cols[i] for i in indices][:10], palette="viridis")
    plt.xlabel("Важность (Importance)")
    plt.show()
    return rf_model, df, y_test, y_prob

    

def generate_prediction_map(model, path_to_tifs, output_file="europe_distribution.tif"):
    
    tif_files = [os.path.join(path_to_tifs, f"wc2.1_10m_bio_{i}.tif") for i in range(1, 20)]
    
    if os.path.exists(tif_files[0]):
        with rasterio.open(tif_files[0]) as src:
            meta = src.meta.copy()
            height, width = src.shape
            transform = src.transform
            nodata_val = src.nodata
            europe_bbox = box(-25.0, 34.0, 45.0, 72.0)
            mask_europe = geometry_mask([europe_bbox], out_shape=(height, width), transform=transform)

        raster_stack = []
        try:
            for f in tif_files:
                with rasterio.open(f) as src:
                    arr = src.read(1)
                    arr = arr.astype('float32')
                    if nodata_val is not None:
                        arr[arr == nodata_val] = np.nan
                    raster_stack.append(arr)
        except Exception as e:
            print(f"Ошибка чтения файла: {e}")

        stack = np.array(raster_stack)
        n_features, h, w = stack.shape
        X_map = stack.reshape(n_features, -1).T
        nan_mask = np.isnan(X_map).any(axis=1)
        europe_mask_flat = mask_europe.reshape(-1)
        valid_pixels_mask = (~nan_mask) & (~europe_mask_flat)
        
        if  valid_pixels_mask.sum() != 0:
            prediction_flat = np.full(h * w, np.nan, dtype='float32')
            X_valid = X_map[valid_pixels_mask]
            probs = model.predict_proba(X_valid)[:, 1]
            prediction_flat[valid_pixels_mask] = probs
            prediction_map = prediction_flat.reshape(h, w)
            meta.update(dtype=rasterio.float32, count=1, nodata=-9999)
            with rasterio.open(output_file, 'w', **meta) as dst:
                dst.write(prediction_map, 1)  
            print(f"Карта сохранена как: {output_file}")
            
            # 7. Быстрая визуализация
            plt.figure(figsize=(10, 8))
            plt.imshow(prediction_map, cmap='Spectral_r', vmin=0, vmax=1)
            plt.colorbar(label="Вероятность обитания")
            plt.title("Species Distribution Model: Europe")
            plt.axis('off')
            plt.show()
        else:
            print("Нет валидных пикселей для предсказания. Проверьте координаты BBOX.")
    else:
        print("ОШИБКА: Не найдены файлы климата. Проверьте путь PATH_TO_WORLDCLIM")




def calculate_spatial_fold_accuracy(df_full, X_data, Y_data, model, block_size=5):
    df_full['block_lat'] = (df_full['lat'] // block_size).astype(int)
    df_full['block_lon'] = (df_full['lon'] // block_size).astype(int)
    df_full['block_id'] = df_full['block_lat'].astype(str) + '_' + df_full['block_lon'].astype(str)
    
    unique_blocks = df_full['block_id'].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_blocks_idx, test_blocks_idx) in enumerate(kf.split(unique_blocks)):
        
        train_blocks = unique_blocks[train_blocks_idx]
        test_blocks = unique_blocks[test_blocks_idx]
        
        train_indices = df_full[df_full['block_id'].isin(train_blocks)].index
        test_indices = df_full[df_full['block_id'].isin(test_blocks)].index
        
        X_train_spatial = X_data.loc[train_indices]
        Y_train_spatial = Y_data.loc[train_indices]
        X_test_spatial = X_data.loc[test_indices]
        Y_test_spatial = Y_data.loc[test_indices]
        
        if len(Y_test_spatial) != 0:
            model.fit(X_train_spatial, Y_train_spatial)
            Y_prob_spatial = model.predict_proba(X_test_spatial)[:, 1]
            
            roc_auc = roc_auc_score(Y_test_spatial, Y_prob_spatial)
            fold_results.append(roc_auc)
            print(f"Фолд {fold+1} (блоки: {len(test_blocks)}), ROC-AUC: {roc_auc:.4f}")
        
    return np.mean(fold_results)


def validation(final_df, y_test, y_prob):
    bio_cols = [f"BIO{i}" for i in range(1, 20)]
    X_clean = final_df[bio_cols]  
    y = final_df['species_present']  
    df_coords = final_df[['lat', 'lon']].copy()  
    df_spatial = pd.concat([df_coords, X_clean, y], axis=1)

   
    mean_spatial_auc = calculate_spatial_fold_accuracy(
        df_spatial.copy(), 
        X_clean, 
        y, 
        RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, class_weight='balanced'),
        block_size=5
    )

    ordinary_auc = roc_auc_score(y_test, y_prob)  

    print("\n--- РЕЗУЛЬТАТ ПРОВЕРКИ АДЕКВАТНОСТИ ---")
    print(f"ROC-AUC при обычной (случайной) валидации: {ordinary_auc:.4f}")
    print(f"Средний ROC-AUC при пространственной (блочной) валидации: {mean_spatial_auc:.4f}")

    if mean_spatial_auc < ordinary_auc * 0.9:  
        print(" ВНИМАНИЕ: ПАДЕНИЕ ТОЧНОСТИ!")
        print(" Модель может быть неадекватна для новых территорий (возможно переобучение).")
    else:
        print(" МОДЕЛЬ АДЕКВАТНА!")

    # Дополнительная визуализация результатов
    plt.figure(figsize=(10, 6))
    methods = ['Обычная валидация', 'Пространственная валидация']
    scores = [ordinary_auc, mean_spatial_auc]
    colors = ['blue', 'green']

    bars = plt.bar(methods, scores, color=colors, alpha=0.7)
    plt.ylabel('ROC-AUC')
    plt.title('Сравнение методов валидации модели')
    plt.ylim(0, 1.0)
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.4f}', ha='center', va='bottom')

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
   i = 2
   data =model()
   generate_prediction_map(data[0], PATH_TO_WORLDCLIM)
   while(i!=0):
    i = int( input("Для создания новой модели введите 1  \nДля валидации модели введите 2 \nДля выхода введите 0 \n"))
    if i ==1:
        data =model()
        generate_prediction_map(data[0], PATH_TO_WORLDCLIM)
    elif(i==2):
        validation(data[1], data[2], data[3])


