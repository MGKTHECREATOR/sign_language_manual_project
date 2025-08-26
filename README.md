
 🖐️ Sign Language Detection System

This project is an **AI-powered system** that detects **hand gestures** from a webcam and translates them into **alphabets, numbers, and words**. It uses **Python, OpenCV, and Mediapipe** for hand tracking, and a **Random Forest model** for classification.

 🚀 Features

* 🎥 Real-time sign detection using webcam
* 🔤 Supports **A–Z alphabets** and **1–9 numbers**
* 📝 Forms **sentences** from multiple sign inputs
* ␣ Includes **Space** and **Clear** functions for text editing
* 🤖 Machine learning model trained on **custom dataset**
* ⚡ Lightweight and runs on CPU

📂 Project Structure

```
sign_language_project/
│── train_model.py        # Train the ML model
│── detect_sign.py        # Run detection with webcam
│── requirements.txt      # Dependencies
│── sign_model.pkl        # Saved trained model
│── dataset/              # Your gesture dataset
│── README.md             # Project documentation
```

 ⚙️ Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/sign-language-detection.git
cd sign-language-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model (if dataset updated):

```bash
python train_model.py
```

4. Run real-time detection:

```bash
python detect_sign.py
```

---

🎯 Usage

* Show a **hand gesture** in front of the webcam
* The system will display the **detected alphabet/number**
* Use gestures repeatedly to form a **sentence**
* Use **Space** (gesture) → adds a space
* Use **Clear** (gesture) → clears text

---
🛠️ Tech Stack

* **Python**
* **OpenCV** (image processing)
* **Mediapipe** (hand landmarks detection)
* **Scikit-learn** (Random Forest Classifier)

---

📌 Future Improvements

* 🔊 Text-to-Speech for spoken output
* 📱 Mobile App integration
* 🌍 Support for regional sign languages

---

🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.

---
📜 License

This project is licensed under the **MIT License** – feel free to use and modify.

