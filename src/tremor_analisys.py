#!/usr/bin/env python3
import os
from importlib.resources import files

import pandas as pd
import pytz
import matplotlib.dates as mdates
from pathlib import Path
import matplotlib.pyplot as plt
import zipfile
import requests
from io import BytesIO
import tempfile
import json
from datetime import datetime
# Paradigma imports
from paradigma.util import load_tsdf_dataframe, write_df_data
from paradigma.config import IMUConfig, TremorConfig
from paradigma.preprocessing import preprocess_imu_data
from paradigma.pipelines.tremor_pipeline import (
    extract_tremor_features_safe,
    detect_tremor
)
import tsdf


def main(patient_id: str):


    # ——— CONFIG —————————————————————————————————————————————
    prefix_raw   = "IMU"
    config_imu   = IMUConfig()

    # URL del tuo API Gateway — endpoint GET che crea lo zip e restituisce direttamente il presigned URL
    target_url = "https://3qpkphed39.execute-api.us-east-1.amazonaws.com/dev/api/inference/getInput"

    def get_input_zip_url(patient_id: str):
        """
        Invoca GET /getInput su API Gateway con header patientid e restituisce il presigned URL.
        """
        headers = {
            "patientid": patient_id
        }
        print("[DEBUG] headers:", headers, flush=True)

        resp = requests.get(target_url, headers=headers)
        resp.raise_for_status()

        payload = resp.json()
        print("[DEBUG] payload:", payload, flush=True)

        url = payload.get("url")
        if not url:
            raise RuntimeError(f"Risposta priva di 'url': {payload}")

        return url
    def download_and_extract_all(tmp_dir, presigned_url):
        """
        Scarica lo zip contenente tutte le cartelle di input/ ed estrae in tmp_dir.
        Restituisce il Path della directory radice con le sottocartelle di batch.
        """
        resp = requests.get(presigned_url)
        resp.raise_for_status()
        with zipfile.ZipFile(BytesIO(resp.content)) as z:
            z.extractall(tmp_dir)

        tmp_path = Path(tmp_dir)
        # Se dopo l'estrazione c'è una singola directory (es. 'input'), usala come root
        subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            return subdirs[0]
        # Altrimenti, usa direttamente tmp_dir
        return tmp_path


    def extract_data_zip_local(data_zip_path, dest_dir):
        if not data_zip_path.exists():
            raise FileNotFoundError(f"{data_zip_path} non esiste")
        dest = dest_dir / "data"
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(data_zip_path, 'r') as z:
            z.extractall(dest)
        return dest

    # ——— MAIN —————————————————————————————————————————————
    # 1) Richiedi a Lambda di creare lo zip e ottenere il presigned URL
    print("Richiedo zip di input/ e presigned URL...")
    print(f"[DEBUG] patient_id ricevuto in main(): {patient_id!r}")
    url_input = get_input_zip_url(patient_id)
    print(f"URL ricevuto: {url_input}")

    # 2) Download ed estrazione del file zip in tmpdir
    df_all_aligned = []
    with tempfile.TemporaryDirectory() as tmpdir:
        print("📥 Download ed estrazione di tutte le cartelle di input/...")
        root_input = download_and_extract_all(tmpdir, url_input)

        # Elaborazione batch per batch
        for batch_dir in sorted(root_input.iterdir()):
            if not batch_dir.is_dir():
                continue
            batch_id = batch_dir.name
            try:
                print(f"🔄 Processing batch: {batch_id}")
                data_zip = batch_dir / "data.zip"
                batch_tmp = Path(tmpdir) / f"batch_{batch_id}"
                extracted_data = extract_data_zip_local(data_zip, batch_tmp)

                df_raw, metadata_time, metadata_values = load_tsdf_dataframe(
                    extracted_data, prefix=prefix_raw
                )

                # Allineamento temporale
                start_dt = pd.to_datetime(metadata_time.start_iso8601)
                df_raw['time_dt'] = start_dt + pd.to_timedelta(df_raw['time'], unit='ms')
                freq_ms = int(1000 / config_imu.sampling_frequency)
                df_aligned = (
                    df_raw.set_index('time_dt')
                          .asfreq(f'{freq_ms}ms')
                          .interpolate(method='time')
                          .reset_index()
                )
                df_aligned['batch_id'] = batch_id
                df_aligned['start_dt'] = start_dt

                df_all_aligned.append(df_aligned)
            except Exception as e:
                print(f"❌ Skipping batch {batch_id} per errore: {e}")
                continue

    # ——— UNIONE DATI —————————————————————————————————————————————
    if not df_all_aligned:
        raise RuntimeError("No data found in any batch folder.")



    df_concat = pd.concat(df_all_aligned, ignore_index=True)

    # ——— ORDINAMENTO & TEMPO RELATIVO CONTIGUO ——————————————————
    df_concat = df_concat.sort_values(by='time_dt').reset_index(drop=True)
    df_concat = df_concat.drop_duplicates(subset='time_dt')
    df_concat = df_concat.dropna(subset=['time_dt'])

    # DEBUG: Intervallo originale nei dati unificati
    print(">>> Intervallo tempo in df_concat (tutti i dati grezzi unificati):")
    print("Start:", df_concat['time_dt'].min())
    print("End:  ", df_concat['time_dt'].max())
    print("Totale righe:", len(df_concat))

    # Ricostruzione tempo relativo contiguo
    expected_step = 1.0 / config_imu.sampling_frequency
    df_concat['time'] = (pd.Series(range(len(df_concat))) * expected_step).round(6)




    # DEBUG
    print(">>> Check tempo ricostruito")
    print("Numero righe:", len(df_concat))
    print("Step previsto:", expected_step)
    print("Min step:", df_concat['time'].diff().min())
    print("Max step:", df_concat['time'].diff().max())

    print("Timestamp di inizio:", df_concat['time_dt'].min())
    print("Timestamp di fine:  ", df_concat['time_dt'].max())



    # ——— PREPROCESSING —————————————————————————————————————————
    # Salva time_dt prima del preprocessing
    time_dt_series = df_concat['time_dt'].reset_index(drop=True)

    # Preprocessing
    df_pre = preprocess_imu_data(df_concat, config_imu, sensor='gyroscope', watch_side='left')
    print(f"Unified dataset resampled to {config_imu.sampling_frequency} Hz.")

    # Aggiunge time_dt di nuovo
    df_pre['time_dt'] = time_dt_series


    print(">>> Intervallo tempo in df_pre (dati preprocessati):")
    print("Start:", df_pre['time_dt'].min())
    print("End:  ", df_pre['time_dt'].max())
    print("Totale righe:", len(df_pre))

    # ——— FEATURE EXTRACTION ————————————————————————————————
    config_feat = TremorConfig(step='features')
    config_feat.window_length_s = 1.0
    config_feat.window_step_length_s = 0.5
    config_feat.segment_length_psd_s = 1
    config_feat.segment_length_spectrogram_s = 1
    config_feat.rest_std_threshold       = 0.005   # default ~0.005; prova anche 0.003 o 0.002
    # • (opzionale) se il tuo pipeline lo supporta, aggiungi anche un vincolo
    #   su peak-to-peak dell’accelerazione per escludere vibrazioni residue:
    config_feat.rest_ptp_threshold       = 0.02    # in g; calibra in base ai tuoi dati

    setattr(config_feat, 'nperseg', 128)

    df_pre["time"] = (df_pre["time_dt"] - df_pre["time_dt"].iloc[0]).dt.total_seconds()
    df_feat = extract_tremor_features_safe(df_pre, config_feat)

    # Ricava time_dt dai secondi relativi
    t0 = df_pre['time_dt'].iloc[0]
    df_feat['time_dt'] = (
        t0 + pd.to_timedelta(df_feat['time'], unit='s')
    ).dt.tz_convert("Europe/Rome")




    print(">>> Intervallo tempo in df_feat (feature estratte):")
    print("Start:", df_feat['time_dt'].min())
    print("End:  ", df_feat['time_dt'].max())
    print("Totale righe:", len(df_feat))

    # ——— CLASSIFICAZIONE ————————————————————————————————
    clf_path = files('paradigma') / 'assets' / 'tremor_detection_clf_package.pkl'
    df_pred = detect_tremor(df_feat.copy(), config_feat, clf_path)


    # calcola threshold dinamicamente
    rest_power_threshold = df_pred['below_tremor_power'].quantile(0.15) #prendi il 15% dei periodi più tranquilli come soglia

    # ricrea il flag: 1 se sotto soglia, 0 altrimenti
    df_pred['pred_arm_at_rest'] = (
        df_pred['below_tremor_power'] < rest_power_threshold
    ).astype('int8').astype('float64')



    print("colonne")
    print(df_pred.columns)
    print(df_pred['below_tremor_power'].describe())



    # ——— QUANTIFICAZIONE ————————————————————————————————
    df_quant = df_pred[
        ['time', 'pred_arm_at_rest', 'pred_tremor_checked', 'tremor_power', 'pred_tremor_proba', 'freq_peak']].copy()
    df_quant['time_dt'] = df_pred['time_dt']
    df_quant['time_dt'] = df_quant['time_dt'].dt.tz_convert("Europe/Rome")





    # ——— SALVATAGGIO RISULTATI IN FORMATO TSDF ———————————————————————————————
    from paradigma.util import write_df_data

    # Percorso di output
    output_dir = Path("inference_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Usa i metadati originali come base
    meta_time_pred = tsdf.TSDFMetadata(
        metadata_time.get_plain_tsdf_dict_copy(), output_dir
    )
    meta_time_pred.channels  = ['time']
    meta_time_pred.units     = ['Relative seconds']
    meta_time_pred.data_type = float
    meta_time_pred.file_name = "IMU_pred_time.bin"

    meta_vals_pred = tsdf.TSDFMetadata(
        metadata_values.get_plain_tsdf_dict_copy(), output_dir
    )
    meta_vals_pred.channels  = [
        'pred_tremor_proba',
        'pred_tremor_checked',
        'pred_arm_at_rest',
        'tremor_power',
        'freq_peak'
    ]
    meta_vals_pred.units     = ['Unitless'] * len(meta_vals_pred.channels)

    # ← qui: fai corrispondere 5 scale_factors, uno per ogni canale
    # se i tuoi dati sono già in unità “finali” metti 1.0, altrimenti i valori di scala che servono
    meta_vals_pred.scale_factors = [1.0] * len(meta_vals_pred.channels)




    meta_vals_pred.data_type = float
    meta_vals_pred.file_name = "IMU_pred_values.bin"

    meta_pred_filename = "IMU_pred_meta.json"

    print("DF types before saving:\n", df_pred.dtypes)


    # ⚠️ Forza conversione a float64 per evitare errori durante salvataggio binario
    cols_to_convert = ['pred_tremor_proba', 'pred_tremor_checked', 'pred_arm_at_rest', 'freq_peak']
    for col in cols_to_convert:
        df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce').astype('float64')

    # ⚠️ Converte il tempo da datetime a secondi relativi per il salvataggio binario
    df_pred_to_save = df_pred.copy()
    # Assumendo df_pred_to_save['time'] sia già in secondi relativi
    df_pred_to_save['time'] = df_pred_to_save['time'] - df_pred_to_save['time'].iloc[0]

    # (Opzionale) Controllo finale
    print("DF types before saving:\n", df_pred_to_save[cols_to_convert + ['time']].dtypes)

    # Scrive i file TSDF
    write_df_data(
        meta_time_pred,
        meta_vals_pred,
        output_dir,
        meta_pred_filename,
        df_pred_to_save[['time', 'pred_tremor_proba', 'pred_tremor_checked', 'pred_arm_at_rest', 'tremor_power','freq_peak']]
    )
    # (Opzionale) Controllo finale
    print("DF types before saving:\n", df_pred[cols_to_convert].dtypes)


    print(f"✅ Predizioni salvate in: {output_dir.resolve()}")







    print(">>> Intervallo temporale in df_quant:")
    print("Start:", df_quant['time_dt'].min())
    print("End:  ", df_quant['time_dt'].max())
    print("Totale righe:", len(df_quant))






    def get_output_upload_url(timestamp_str: str, patient_id:str):
        """
        Richiede a Lambda un presigned URL per caricare predictions_<timestamp>.zip
        """
        api_url = "https://3qpkphed39.execute-api.us-east-1.amazonaws.com/dev/api/inference/getOutputUrl"
        filename = f"predictions_{timestamp_str}.zip"

        headers = {
            "Content-Type": "application/json",
            "patientid": patient_id

        }
        payload = {"filename": filename}

        resp = requests.post(api_url, json=payload, headers=headers)
        resp.raise_for_status()

        payload = resp.json()
        body = json.loads(payload.get("body", "{}"))
        url = body.get("url")
        key = body.get("key")

        if not url:
            raise RuntimeError(f"URL mancante nella risposta Lambda: {body}")
        return url, filename

    def upload_and_cleanup(output_dir: Path, timestamp_str: str,patient_id:str):
        """
        Esegue l’upload del file predictions.zip su S3 usando un presigned URL
        e poi elimina il file zip locale (e opzionalmente la cartella di output).
        """
        filename = f"predictions_{timestamp_str}.zip"
        ezip_path = output_dir / filename

        try:
            # 1) Recupera presigned URL per upload
            print("Richiedo presigned URL per l'upload...")
            upload_url, filename_on_s3 = get_output_upload_url(timestamp_now,patient_id)
            print(f"URL di upload ricevuto: {upload_url}")

            # 2) Esegui PUT verso S3
            with open(ezip_path, 'rb') as f:
                print(f"Caricamento di {ezip_path.name} su S3...")
                resp = requests.put(
                    upload_url,
                    data=f,
                    headers={"Content-Type": "application/zip"}
                )
                resp.raise_for_status()

            # 3) Pulizia: elimina zip locale
            print(f"Elimino file locale {ezip_path.name}...")
            os.remove(ezip_path)
            print("✅ File locale eliminato.")

            # (Opzionale) Rimuovi anche la directory di  se non serve più
            # import shutil
            # shutil.rmtree(output_dir)

        except Exception as e:
            print(f"❌ Errore durante l'upload o pulizia: {e}")


    #-------Salvataggio come zip----------
    timestamp_now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"predictions_{timestamp_now}.zip"
    zip_path = output_dir / filename
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fname in [
            meta_time_pred.file_name,
            meta_vals_pred.file_name,
            meta_pred_filename
        ]:
            file_path = output_dir / fname
            if file_path.exists():
                # arcname mantiene solo il nome del file dentro lo ZIP
                z.write(file_path, arcname=fname)
            else:
                print(f"WARNING: {file_path} non trovato, saltato dallo ZIP")


    upload_and_cleanup(output_dir,timestamp_now,patient_id)

    print(f"✅ Tutti i file sono stati compressi in: {zip_path.resolve()}")

    # ——— VISUALIZZAZIONE —————————————————————————————————————
    def visualize(df_q):
        # Tremor Power
        plt.figure(figsize=(8, 4))
        plt.plot(df_q['time_dt'], df_q['tremor_power'], marker='o')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=pytz.timezone("Europe/Rome")))  # 👈 questo!
        plt.xlabel("Time (CET)")
        plt.ylabel("Tremor Power")
        plt.title("Tremor Power over Time")
        plt.tight_layout()
        plt.show()

        # Tremor Probability
        plt.figure(figsize=(8, 4))
        plt.plot(df_q['time_dt'], df_q['pred_tremor_proba'], marker='o')
        plt.xlabel("Time (CET)")
        plt.ylabel("Tremor Probability")
        plt.title("Tremor Probability over Time")
        ylim_max = df_q['pred_tremor_proba'].max() * 1.2 if df_q['pred_tremor_proba'].max() > 0 else 1
        plt.ylim(0, ylim_max)
        plt.tight_layout()
        plt.show()

        # Arm at Rest
        plt.figure(figsize=(8, 2))
        plt.scatter(df_q[df_q['pred_arm_at_rest'] == 1]['time_dt'],
                    df_q[df_q['pred_arm_at_rest'] == 1]['pred_arm_at_rest'],
                    s=10)
        plt.title("Arm at Rest Over Time")
        plt.xlabel("Time (CET)")
        plt.ylabel("Rest Flag")
        plt.tight_layout()
        plt.show()

        # Peak Frequency
        plt.figure(figsize=(8, 4))
        plt.plot(df_q['time_dt'], df_q['freq_peak'], marker='o')
        plt.axhline(y=3.0, color='gray', linestyle='--', label='Lower Bound 3 Hz')
        plt.axhline(y=7.0, color='gray', linestyle='--', label='Upper Bound 7 Hz')
        plt.xlabel("Time (CET)")
        plt.ylabel("Peak Frequency (Hz)")
        plt.title("Peak Frequency vs. Time")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("Visualizing combined quantification across all batches...")
    visualize(df_quant)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Errore: patient_id mancante")
        sys.exit(1)
    patient_id = sys.argv[1]
    main(patient_id)

