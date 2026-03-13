#  AI Predictive Maintenance: Vibration Analysis & Forecasting

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Prophet](https://img.shields.io/badge/Model-Prophet-FF9900.svg)
![Matplotlib](https://img.shields.io/badge/Data_Viz-Matplotlib-green.svg)

โปรเจกต์วิเคราะห์และพยากรณ์ความสั่นสะเทือนของเครื่องจักร (Vibration Analysis) ด้วยเทคโนโลยี **Machine Learning (Time-Series Forecasting)** เพื่อทำ **Predictive Maintenance** (การบำรุงรักษาเชิงคาดการณ์) ช่วยป้องกันเครื่องจักรพังเสียหายก่อนเวลาอันควร และประเมินความรุนแรงตามมาตรฐาน **ISO 10816-3**

##  ฟีเจอร์หลัก (Key Features)

1. **Custom Raw Data Parser:** ระบบสกัดข้อมูลอัจฉริยะ สามารถอ่านไฟล์ `.txt` ดิบจากเซ็นเซอร์ (Waveform Amplitudes) ที่มีการบันทึกแบบหลายคอลัมน์ซ้อนกัน ข้ามข้อมูลขยะ และดึงเฉพาะค่า Peak ของแต่ละวันออกมาได้อัตโนมัติ
2. **Text Normalization:** ระบบกรองคำและจัดกลุ่มชื่อเครื่องจักรอัตโนมัติ (เช่น รวบรวม `(CHPP) Cooling Pump for OAH-02` และ `Cooling Pump for OAH-02` ให้เป็นเครื่องเดียวกัน)
3. **AI Forecasting (Prophet):** ใช้โมเดลพยากรณ์อนุกรมเวลา (Time-Series) เพื่อคาดการณ์ความสั่นสะเทือนล่วงหน้า 30 วัน
4. **Automated ISO 10816-3 Alerting:** สร้างกราฟเส้น (Line Chart) อัตโนมัติ พร้อมประเมินสถานะ 🟢 Normal, 🟡 Warning, 🟠 Alert ตามเกณฑ์มาตรฐานวิศวกรรมสากล

##  โครงสร้างโฟลเดอร์ (Directory Structure)
```text
predictive-maintenance-ai/
│
├── data/                       # โฟลเดอร์เก็บไฟล์ .txt ข้อมูลสั่นสะเทือน (Waveform)
│   ├── A_CH-06...txt
│   ├── A_Cooling_Pump...txt
│   └── A_Jockey_pump...txt
│
├── train_model.py              # สคริปต์หลักสำหรับประมวลผลและสร้างกราฟพยากรณ์
├── requirements.txt            # รายชื่อไลบรารีที่ต้องใช้ (Dependencies)
└── README.md                   # เอกสารอธิบายโปรเจกต์
```
