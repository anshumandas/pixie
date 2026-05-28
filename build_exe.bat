@echo off
REM ===========================================================================
REM  Build Pixie.exe on Windows.
REM  Double-click this file, or run it from a terminal in the project folder.
REM ===========================================================================
setlocal

echo [1/4] Creating / activating virtual environment...
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate

echo [2/4] Installing app dependencies...
pip install -r requirements.txt

echo [3/4] Installing PyInstaller...
pip install pyinstaller

echo [4/4] Building...
pyinstaller pixie.spec --noconfirm

echo.
echo ============================================================
echo  Done.  Your app is here:  dist\Pixie\Pixie.exe
echo  (zip the dist\Pixie folder to share it)
echo ============================================================
pause
