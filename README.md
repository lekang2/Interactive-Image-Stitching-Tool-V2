# Interactive-Image-Stitching-Tool-V2

An interactive Django web app for stitching images using **SIFT + RANSAC**, 
with statistical validation through hypothesis testing.  

The application dynamically visualizes:
- Feature points (scatter plots)
- Descriptor histograms
- Network graphs of matches
- Final stitched image (when validated)

- ## 🎥 Demo


## 🌟 Features
- Upload two images via a clean Django interface
- Extract SIFT features and compute 128-d descriptors
- Match features and reject outliers with RANSAC
- Hypothesis testing (t-test) for stitchability
- Dynamic visualizations of feature points and matches
- Dockerized deployment on AWS EC2

---

## 🚀 Quick Start

### Local Run
```bash
git clone https://github.com/lekang2/Interactive-Image-Stitching-Tool-V2.git
cd Interactive-Image-Stitching-Tool-V2/src
pip install -r requirements.txt
python manage.py runserver

Then open http://127.0.0.1:8000 in your browser.

Docker Deployment
cd docker
docker build -t image-stitching .
docker run -p 8000:8000 image-stitching

📂 Project Structure

src/ – Django web application

algorithms/ – Computer vision modules (SIFT, RANSAC, hypothesis test)

evaluation/ – Visualization scripts and results

docker/ – Deployment files

docs/ – Reports, presentation, architecture, demo video
