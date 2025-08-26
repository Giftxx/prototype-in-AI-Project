# 🖼️ Image Processing Dashboard

แอปพลิเคชัน Streamlit สำหรับ Image Processing ที่มีฟีเจอร์ครบครัน

## ✨ Features

### 📷 Input Sources
1. **Upload File** - อัปโหลดไฟล์รูปภาพ (PNG, JPG, JPEG, BMP, TIFF)
2. **Camera** - ถ่ายภาพจากกล้องเว็บแคม
3. **URL** - โหลดรูปภาพจาก URL บนอินเทอร์เน็ต

### 🔧 Basic Operations (Non-Parameter)
1. **ROI (Region of Interest)** - ครอปภาพเฉพาะส่วนที่สนใจ
2. **Before/After Viewer** - แสดงเปรียบเทียบภาพก่อนและหลังการประมวลผล
3. **Grayscale** - แปลงภาพเป็นโทนสีเทา

### ⚙️ Advanced Processing (Parameter-based)
1. **Color Space Conversion**
   - HSV (Hue, Saturation, Value)
   - LAB (L*a*b* color space)

2. **Edge Detection**
   - Canny Edge Detection
   - ปรับแต่ง Low/High Threshold

3. **Gaussian Blur**
   - ปรับ Kernel Size และ Sigma X

4. **Pencil Sketch**
   - สร้างเอฟเฟกต์การวาดดินสอ

5. **Adaptive Thresholding**
   - ปรับแต่ง Max Value, Adaptive Method, Block Size, C Value

### 📊 Analytics & Visualization
1. **Image Metrics Dashboard**
   - ขนาดรูปภาพ (Dimensions)
   - ขนาดไฟล์ (File Size)
   - จำนวนช่องสี (Channels)
   - จำนวนพิกเซล (Total Pixels)

2. **Image Analysis Charts**
   - Histogram Analysis - แสดงการกระจายตัวของสีในภาพ
   - Color Channel Distribution (Pie Chart)
   - Image Statistics (Bar Chart)

3. **Detailed Statistics Table**
   - สถิติรายละเอียดของภาพ

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
python -m streamlit run app3.py
```

หรือ

```bash
streamlit run app3.py
```

### Access the App
เปิดเบราว์เซอร์และไปที่: `http://localhost:8501`

## 📋 Requirements
- streamlit
- opencv-python
- numpy
- matplotlib
- pandas
- pillow
- plotly
- requests
- urllib3

## 🎯 How to Use

1. **เลือกวิธีการนำเข้าภาพ** จาก Sidebar
   - อัปโหลดไฟล์
   - ถ่ายภาพจากกล้อง
   - ใส่ URL ภาพ

2. **เลือกการประมวลผลที่ต้องการ**
   - Basic Operations (ไม่ต้องปรับค่า)
   - Advanced Processing (ปรับค่าพารามิเตอร์ได้)

3. **ดูผลลัพธ์**
   - ภาพที่ประมวลผลแล้ว
   - เมตริกและสถิติ
   - กราฟการวิเคราะห์

## 🎨 UI Design
แอปพลิเคชันออกแบบด้วย:
- **Dark Theme** - สีมืดที่ดูทันสมัย
- **Responsive Layout** - ปรับขนาดได้ตามหน้าจอ
- **Interactive Widgets** - ปรับแต่งค่าได้แบบ Real-time
- **Professional Dashboard** - แสดงข้อมูลแบบ Dashboard

## 📸 Screenshots
แอปจะแสดง:
- Header ที่สวยงามพร้อม Gradient
- Sidebar สำหรับควบคุม
- Metric Cards แสดงข้อมูลสำคัญ
- กราฟและชาร์ตแบบ Interactive

## 🔧 Technical Details
- **Frontend**: Streamlit
- **Image Processing**: OpenCV
- **Visualization**: Plotly, Matplotlib
- **Data Handling**: Pandas, NumPy
- **Styling**: Custom CSS

---
**Created with ❤️ using Streamlit**
