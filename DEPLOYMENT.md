# 🚀 Free Deployment Guide: Label Audit

This project can be deployed **100% free with zero monthly charges** using any of the following platforms.

---

## 🌟 Option 1: Hugging Face Spaces (Recommended for Machine Learning)
**Why it's best**:
- Free 16 GB RAM + 2 vCPU forever (more than enough for the 1 GB XLM-RoBERTa model).
- Free public `https://<your-username>-label-audit.hf.space` URL.
- Free SSL certificate included.

### Steps:
1. Go to [huggingface.co](https://huggingface.co/) and sign up / log in (free).
2. Click **New Space** (`https://huggingface.co/new-space`).
3. Set:
   - **Space name**: `label-audit`
   - **License**: `mit`
   - **Space SDK**: Choose **Docker** -> **Blank**.
   - **Space hardware**: Free `2 vCPU, 16 GB RAM`.
4. Clone the space or push this project folder:
   ```bash
   git remote add space https://huggingface.co/spaces/<YOUR-USERNAME>/label-audit
   git add .
   git commit -m "Deploy Label Audit"
   git push space main
   ```
5. Hugging Face will automatically build the `Dockerfile` and give you a public URL you can share with anyone!

---

## 🌐 Option 2: Render.com (Free Web Service)
**Why it's great**:
- Generates a free `https://label-audit.onrender.com` URL.
- Connects directly to GitHub.

### Steps:
1. Push this folder to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Deploy Label Audit"
   git remote add origin https://github.com/<YOUR-USERNAME>/label-audit.git
   git push -u origin main
   ```
2. Log into [render.com](https://render.com) with GitHub.
3. Click **New +** -> **Web Service**.
4. Select your `label-audit` repository.
5. Set:
   - **Environment**: `Docker`
   - **Instance Type**: `Free`
6. Click **Create Web Service**. Done!

---

## ⚡ Option 3: Instant Public Demo Right Now (via Ngrok / LocalTunnel)
If you need an immediate free public link to share with an examiner or colleague **today while running on your laptop**:

### Using LocalTunnel (No account needed):
```powershell
npx localtunnel --port 8050
```
It gives you an instant public link: `https://<random-name>.loca.lt` pointing directly to your running server!

### Using Ngrok:
```powershell
ngrok http 8050
```
