Personal Fitness Tracker & Coach Backend (FastAPI version)
=======================================================

This module provides a FastAPI web application that serves both a REST API and
HTML pages for a simple, self‑hosted fitness tracking and coaching tool.
It persists data in SQLite, exposes endpoints for managing workouts,
scheduling, logging completed sessions, and obtaining AI recommendations via
OpenAI's API. Jinja2 is used to render the HTML templates.

The app is designed to run on a home server and be accessed from an
iPhone or other web browsers. To start the server:

    $ export OPENAI_API_KEY=sk-...
    $ uvicorn workout_app.server:app --host 0.0.0.0 --port 5000 --reload

Note: The `uvicorn` command is available through the `uvicorn` package
preinstalled in this environment. If running behind a reverse proxy, ensure
proper HTTPS configuration for security when exposing the server publicly.
"""

import os
import json
import datetime
import sqlite3
from typing import Dict, List

import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


DATABASE = os.path.join(os.path.dirname(__file__), 'workouts.db')



def get_db_connection() -> sqlite3.Connection:
    """Return a new database connection with a Row factory for dict‑like access."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn



def init_db() -> None:
    """Initialize the database if it doesn't already exist."""
    with get_db_connection() as db:
        # workouts table holds definitions of both strength and cardio exercises
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS workouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL CHECK (type IN ('strength', 'cardio')),
                sets INTEGER,
                reps INTEGER,
                weight REAL,
                distance REAL,
                duration REAL
            )
            """
        )
        # workout_logs table holds completed sessions
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS workout_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workout_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                log_data TEXT NOT NULL,
                comment TEXT,
                FOREIGN KEY (workout_id) REFERENCES workouts (id) ON DELETE CASCADE
            )
            """
        )
        # schedule table maps dates to workouts
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                workout_id INTEGER NOT NULL,
                FOREIGN KEY (workout_id) REFERENCES workouts (id) ON DELETE CASCADE
            )
            """
        )
        db.commit()



def fetch_ai_recommendation(history: List[Dict]) -> str:
    """Request an AI recommendation from OpenAI based on workout history.

    See the Flask version of this function for details. It constructs a
    structured prompt and calls the Chat Completions API. The API key must
    be supplied via the OPENAI_API_KEY environment variable.
    """
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return (
            "OpenAI API key not set. Please set the OPENAI_API_KEY environment"
            " variable to receive recommendations."
        )
    system_prompt = (
        "You are a helpful personal training assistant. Your job is to analyse "
        "workout history and suggest when to increase weight or adjust volume. "
        "Provide succinct, actionable advice tailored to the user's recent "
        "performance."
    )
    history_snippet = []
    for entry in history[-10:]:
        w_type = entry.get('type')
        if w_type == 'strength':
            sets = entry.get('sets_completed', [])
            sets_str = ", ".join([
                f"{s['reps']} reps @ {s['weight']}kg" for s in sets
            ])
            history_snippet.append(
                f"On {entry['date']} you performed {entry['name']} with sets: {sets_str}."
            )
        elif w_type == 'cardio':
            history_snippet.append(
                f"On {entry['date']} you ran {entry.get('distance', '?')} km in "
                f"{entry.get('duration', '?')} minutes for the workout {entry['name']}."
            )
        else:
            history_snippet.append(
                f"On {entry['date']} you completed {entry['name']}."
            )
    user_message = (
        "Here is my recent workout history:\n" + "\n".join(history_snippet) +
        "\nBased on this, please recommend whether I should increase the weight "
        "or intensity for each exercise, and provide suggestions for progression "
        "in both strength and running workouts."
    )
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.5,
        "max_tokens": 200,
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if data.get('choices'):
            return data['choices'][0]['message']['content'].strip()
        return "No recommendation available."
    except Exception as exc:
        return f"Error requesting recommendation: {exc}"


# Initialize DB once
init_db()

# Create FastAPI app
app = FastAPI()

# Mount static files (CSS/JS)
app.mount('/static', StaticFiles(directory=os.path.join(os.path.dirname(__file__), 'static')), name='static')

# Configure templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), 'templates'))


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.get('/workouts', response_class=HTMLResponse)
async def workouts_page(request: Request):
    return templates.TemplateResponse('workouts.html', {'request': request})


@app.get('/schedule', response_class=HTMLResponse)
async def schedule_page(request: Request):
    return templates.TemplateResponse('schedule.html', {'request': request})


@app.get('/history', response_class=HTMLResponse)
async def history_page(request: Request):
    return templates.TemplateResponse('history.html', {'request': request})


# API endpoints

@app.get('/api/workouts')
async def api_get_workouts() -> List[Dict]:
    conn = get_db_connection()
    rows = conn.execute('SELECT * FROM workouts').fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.post('/api/workouts')
async def api_create_workout(workout: Dict) -> Dict:
    required = {'name', 'type'}
    if not required.issubset(workout.keys()):
        raise HTTPException(status_code=400, detail='Invalid workout definition')
    name = workout['name']
    wtype = workout['type']
    if wtype not in ('strength', 'cardio'):
        raise HTTPException(status_code=400, detail='Type must be strength or cardio')
    sets = workout.get('sets')
    reps = workout.get('reps')
    weight = workout.get('weight')
    distance = workout.get('distance')
    duration = workout.get('duration')
    conn = get_db_connection()
    with conn:
        cur = conn.execute(
            'INSERT INTO workouts (name, type, sets, reps, weight, distance, duration) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)',
            (name, wtype, sets, reps, weight, distance, duration),
        )
        new_id = cur.lastrowid
    conn.close()
    return {'id': new_id}


@app.get('/api/workouts/{workout_id}')
async def api_get_workout(workout_id: int):
    conn = get_db_connection()
    row = conn.execute('SELECT * FROM workouts WHERE id=?', (workout_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail='Workout not found')
    return dict(row)


@app.put('/api/workouts/{workout_id}')
async def api_update_workout(workout_id: int, updates: Dict):
    allowed_keys = {'name', 'type', 'sets', 'reps', 'weight', 'distance', 'duration'}
    if not any(k in updates for k in allowed_keys):
        raise HTTPException(status_code=400, detail='No valid fields provided')
    conn = get_db_connection()
    fields = []
    values = []
    for key in allowed_keys:
        if key in updates:
            fields.append(f"{key}=?")
            values.append(updates[key])
    values.append(workout_id)
    with conn:
        conn.execute(f"UPDATE workouts SET {', '.join(fields)} WHERE id=?", tuple(values))
    conn.close()
    return {'status': 'updated'}


@app.delete('/api/workouts/{workout_id}')
async def api_delete_workout(workout_id: int):
    conn = get_db_connection()
    with conn:
        conn.execute('DELETE FROM workouts WHERE id=?', (workout_id,))
    conn.close()
    return {'status': 'deleted'}


@app.get('/api/schedule')
async def api_get_schedule() -> List[Dict]:
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT schedule.id, schedule.date, schedule.workout_id, workouts.name, workouts.type, '
        'workouts.sets, workouts.reps, workouts.weight '
        'FROM schedule JOIN workouts ON schedule.workout_id = workouts.id '
        'ORDER BY schedule.date'
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.post('/api/schedule')
async def api_set_schedule(entries: List[Dict]):
    # entries should be list of {date, workout_id}
    conn = get_db_connection()
    with conn:
        dates = [e['date'] for e in entries if 'date' in e]
        if dates:
            placeholders = ','.join('?' for _ in dates)
            conn.execute(f'DELETE FROM schedule WHERE date IN ({placeholders})', tuple(dates))
        for entry in entries:
            date_str = entry.get('date')
            workout_id = entry.get('workout_id')
            if date_str and workout_id:
                conn.execute('INSERT INTO schedule (date, workout_id) VALUES (?, ?)', (date_str, workout_id))
    conn.close()
    return {'status': 'scheduled'}


@app.delete('/api/schedule')
async def api_clear_schedule():
    conn = get_db_connection()
    with conn:
        conn.execute('DELETE FROM schedule')
    conn.close()
    return {'status': 'cleared'}


@app.get('/api/logs')
async def api_get_logs() -> List[Dict]:
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT workout_logs.id, workout_logs.date, workout_logs.log_data, workout_logs.comment, '
        'workouts.name, workouts.type '
        'FROM workout_logs JOIN workouts ON workout_logs.workout_id = workouts.id '
        'ORDER BY workout_logs.date DESC'
    ).fetchall()
    conn.close()
    results = []
    for row in rows:
        entry = dict(row)
        try:
            entry['log_data'] = json.loads(entry['log_data'])
        except Exception:
            pass
        results.append(entry)
    return results


@app.post('/api/logs')
async def api_create_log(log_entry: Dict):
    workout_id = log_entry.get('workout_id')
    log_data = log_entry.get('log_data')
    comment = log_entry.get('comment')
    date_str = log_entry.get('date') or datetime.date.today().isoformat()
    if not workout_id or not isinstance(log_data, dict):
        raise HTTPException(status_code=400, detail='Invalid log entry')
    conn = get_db_connection()
    with conn:
        conn.execute(
            'INSERT INTO workout_logs (workout_id, date, log_data, comment) VALUES (?, ?, ?, ?)',
            (workout_id, date_str, json.dumps(log_data), comment),
        )
    conn.close()
    return {'status': 'logged'}


@app.get('/api/recommendation')
async def api_get_recommendation():
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT workout_logs.date, workouts.name, workouts.type, workout_logs.log_data '
        'FROM workout_logs JOIN workouts ON workout_logs.workout_id = workouts.id '
        'ORDER BY workout_logs.date DESC LIMIT 10'
    ).fetchall()
    conn.close()
    history_data = []
    for row in reversed(rows):
        entry = {
            'date': row['date'],
            'name': row['name'],
            'type': row['type'],
        }
        try:
            log_info = json.loads(row['log_data'])
            if row['type'] == 'strength':
                entry['sets_completed'] = log_info.get('sets_completed', [])
            elif row['type'] == 'cardio':
                entry['distance'] = log_info.get('distance')
                entry['duration'] = log_info.get('duration')
        except Exception:
            pass
        history_data.append(entry)
    recommendation = fetch_ai_recommendation(history_data)
    return {'recommendation': recommendation}
