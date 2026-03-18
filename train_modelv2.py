import os
import glob
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# ขั้นที่ 1: สร้างเครื่องสกัดข้อมูล (Custom Data Parser)
# ==========================================
def extract_data_from_txt(filepath):
    machine_name = "Unknown"
    record_date = None
    amplitudes = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
        
    for line in lines:
        line = line.strip()
        
        # 1. หาและจัดกลุ่มชื่อเครื่องจักร
        if line.startswith("Equipment:"):
            raw_name = line.split("Equipment:")[1].strip().lower()
            if 'ch-06' in raw_name or 'oah-06' in raw_name:
                machine_name = 'CH-06'
            elif 'cooling pump' in raw_name:
                machine_name = 'Cooling Pump OAH 02'
            elif 'jockey pump' in raw_name:
                machine_name = 'Jockey Pump'
            else:
                machine_name = line.split("Equipment:")[1].strip()
                
        # 2. หาวันที่และเวลา
        elif line.startswith("Date/Time:"):
            date_str = line.split("Date/Time:")[1].split("Amplitude:")[0].strip()
            try:
                record_date = pd.to_datetime(date_str, format="%d-%b-%y %H:%M:%S")
            except:
                pass
                
        # 3. หาข้อมูลตัวเลข
        else:
            parts = line.split()
            if len(parts) >= 2 and (parts[0].replace('.', '', 1).replace('-', '', 1).isdigit()):
                for i in range(1, len(parts), 2):
                    try:
                        amp = float(parts[i])
                        amplitudes.append(abs(amp)) 
                    except ValueError:
                        continue

    if record_date is not None and len(amplitudes) > 0:
        max_amplitude = np.max(amplitudes)
        return machine_name, record_date, max_amplitude
    return None, None, None

# ==========================================
# ขั้นที่ 2: โหลดข้อมูลทั้งหมดเข้า Pandas DataFrame
# ==========================================
print("⏳ กำลังสแกนและสกัดข้อมูลจากไฟล์ .txt ทั้งหมด...")
folder_name = 'data' 
file_list = glob.glob(os.path.join(folder_name, '*.txt'))

if not file_list:
    print(f"❌ ไม่พบไฟล์ .txt ในโฟลเดอร์ '{folder_name}'")
    exit()

extracted_data = []
for file in file_list:
    m_name, dt, max_amp = extract_data_from_txt(file)
    if m_name and dt:
        extracted_data.append([m_name, dt, max_amp])

df = pd.DataFrame(extracted_data, columns=['Machine', 'ds', 'y'])
df = df.sort_values(by=['Machine', 'ds']) 
print(f"✅ สกัดข้อมูลสำเร็จ! ได้ข้อมูลพร้อมเทรนทั้งหมด {len(df)} จุด (จาก {len(file_list)} ไฟล์)\n")

# ==========================================
# 💡 ใหม่! รับค่า "จำนวนวัน" จากผู้ใช้งาน
# ==========================================
print("="*60)
try:
    user_input = input("🎯 ต้องการให้ AI พยากรณ์ล่วงหน้ากี่วัน? (กด Enter เพื่อใช้ค่าเริ่มต้น 30 วัน): ")
    if user_input.strip() == "":
        forecast_days = 30
    else:
        forecast_days = int(user_input)
except ValueError:
    print("⚠️ คุณกรอกข้อมูลไม่เป็นตัวเลข ระบบจะใช้ค่าเริ่มต้นที่ 30 วัน")
    forecast_days = 30
    
print(f"🤖 AI จะทำการพยากรณ์ล่วงหน้า {forecast_days} วัน")
print("="*60 + "\n")

# ==========================================
# ขั้นที่ 3: เทรน AI (Prophet) และสร้างกราฟพยากรณ์
# ==========================================
def check_iso_status(amplitude):
    if amplitude <= 1.4: return "🟢 Normal"
    elif amplitude <= 2.8: return "🟡 Warning"
    elif amplitude <= 4.5: return "🟠 Alert"
    else: return "🔴 Danger"

machine_list = df['Machine'].unique()

for machine in machine_list:
    print(f"⚙️ กำลังประมวลผลโมเดลสำหรับ: {machine}")
    
    df_machine = df[df['Machine'] == machine].copy()
    
    if len(df_machine) < 2:
        print(f"⚠️ ข้อมูลมีน้อยเกินไป ข้ามการทำนาย\n")
        continue
    
    model = Prophet(yearly_seasonality=False, daily_seasonality=False)
    model.fit(df_machine)
    
    # 💡 ใช้ตัวแปร forecast_days ที่รับมาจากผู้ใช้
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    last_day = forecast.iloc[-1]
    pred_date = last_day['ds'].strftime('%Y-%m-%d')
    pred_value = last_day['yhat']
    
    print(f"📈 ทำนายวันที่ {pred_date} | Amplitude: {pred_value:.3f} | สถานะ: {check_iso_status(pred_value)}\n")
    
    # ------------------------------------------
    # พล็อตกราฟเส้น (Line Chart)
    # ------------------------------------------
    plt.figure(figsize=(10, 6))
    
    plt.plot(df_machine['ds'], df_machine['y'], marker='o', markersize=8, linestyle='-', linewidth=2, color='royalblue', label='Historical Data (Actual)')
    
    max_hist_date = df_machine['ds'].max()
    future_forecast = forecast[forecast['ds'] >= max_hist_date]
    
    # 💡 อัปเดตข้อความอธิบายกราฟให้ตรงกับจำนวนวัน
    plt.plot(future_forecast['ds'], future_forecast['yhat'], marker='X', markersize=4, linestyle='--', linewidth=2, color='crimson', label=f'AI Forecast (Next {forecast_days} Days)')
    
    plt.title(f'Vibration Trend & Forecast - {machine}', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Amplitude (Max Peak)', fontsize=12)
    
    plt.axhline(y=1.4, color='forestgreen', linestyle='--', alpha=0.7, linewidth=1.5, label='Normal Limit (1.4)')
    plt.axhline(y=2.8, color='gold', linestyle='--', alpha=0.7, linewidth=1.5, label='Warning Limit (2.8)')
    plt.axhline(y=4.5, color='darkorange', linestyle='--', alpha=0.7, linewidth=1.5, label='Alert Limit (4.5)')
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    
    safe_m_name = machine.replace('/', '-').replace('\\', '-') 
    save_filename = f"LineGraph_Forecast_{safe_m_name}.png"
    plt.savefig(save_filename, bbox_inches='tight', dpi=300)
    plt.close()

print("🎉 เสร็จสมบูรณ์! ลองตรวจสอบไฟล์รูปกราฟในโฟลเดอร์ได้เลยครับ")