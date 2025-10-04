## Getting Started

Disclaimer: For better and more consistent performance, an NVIDIA GPU is recommended while also installing latest Cuda Toolkit.
Will run on device CPU otherwise.
This will take some time to get started at first.

## Run the development server:

```
powershell 1 (Backend)

1.
cd .\py-backend\

2.
py -m venv .venv

3.
.\.venv\Scripts\activate

4.
pip install -r requirements.txt

4.1(Only if host device has Cuda Toolkit installed & has eligible NVIDIA GPU; If not then skip this)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

5.
.\run.bat
```

## Run the Frontend server:

```
powershell 2 (Frontend)

npm install

npm run dev
```

## Once everything is set up

Open [http://localhost:3000](http://localhost:3000) with your browser to run Scout AI.
