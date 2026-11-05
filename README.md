<div align="center">

  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=header&text=SEMI-VISION%20AI&fontSize=40&fontColor=fff&animation=fadeIn&fontAlignY=35" width="100%" />

  <h3>🔬 AI-Powered Semiconductor Image Restoration & Inspection Suite</h3>

  <!-- Animated & Interactive Status Badges -->
  <p>
    <a href="https://github.com/Ganu0124/semiconductor-image-restoration"><img src="https://img.shields.io/github/stars/Ganu0124/semiconductor-image-restoration?style=for-the-badge&logo=github&color=blue" alt="GitHub Stars"/></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
    <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/></a>
    <a href="https://react.dev/"><img src="https://img.shields.io/badge/React-18%2B-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React"/></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License"/></a>
  </p>

  <p>
    <b>An industrial-grade, end-to-end computer vision and deep learning system built for automated semiconductor die, via, and grid inspection, degradation modeling, image restoration, and rigorous metric evaluation.</b>
  </p>

  <p>
    <b>Author:</b> Ganesh M &bull; <b>Contact:</b> <a href="mailto:gganu8615@gmail.com">gganu8615@gmail.com</a>
  </p>

</div>

---

## ⚡ System Architecture & Pipeline Workflow

<div align="center">
  <img src="https://img.shields.io/badge/Status-Active_Pipeline-success?style=for-the-badge&logo=opsgenie&logoColor=white" alt="Status"/>
  <img src="https://img.shields.io/badge/Execution-End_to_End-blueviolet?style=for-the-badge&logo=databricks&logoColor=white" alt="Execution"/>
</div>

```text
[ Raw / Synthetic Dataset ] 
         │
         ▼
[ Configurable Degradation Engine ] ──(Noise, Blur, Downsampling)
         │
         ▼
[ Deep Learning Restoration ] ───────► [ U-Net Baseline ] & [ SwinIR-Lite (Transformer) ]
         │
         ▼
[ Rigorous Metrics Evaluation ] ─────► PSNR | SSIM | LPIPS | MAE | MSE
         │
         ▼
[ FastAPI Backend + SQLite ] ────────► REST Endpoints & Async Processing
         │
         ▼
[ React + Vite Dashboard ] ──────────► Interactive UI & Real-Time Inspection
         │
         ▼
[ ReportLab PDF Generator ] ─────────► Automated Exportable PDF Inspection Reports
