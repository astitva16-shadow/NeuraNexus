# NeuraNexus

NeuraNexus is a Streamlit demo app for brainwave-assisted mental wellness monitoring.

## Features

- Login / signup (SQLite)
- EEG simulation (Theta/Alpha/Beta) + rule-based stress detection
- Alerts + emergency contact workflow (simulated)
- Appointment & consultation booking (SQLite)
  - Book Appointment form
  - My Appointments table (cancel / mark completed)
  - Smart prompts when stress is HIGH

## Run locally

```bash
cd /home/astryx/Desktop/PBL
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

If `streamlit` is not on your PATH, run:

```bash
.venv/bin/python -m streamlit run app.py
```

## Database

The app uses SQLite and creates a local DB file on first run:

- `neuraneux.db`

### Appointments table

The booking system stores data in:

- `appointments(id, user_email, date, time_slot, consultation_type, notes, status)`

Status values:

- `Scheduled`, `Completed`, `Cancelled`
