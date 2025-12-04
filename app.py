"""Dashboard UI redesign with glassmorphism and animations (Dashboard professionalism required)

"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import os
import yaml
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'  # Change this in production

# Configuration
UPLOAD_FOLDER = 'static/uploads'
DATA_FOLDER = 'data'
CONFIG_FOLDER = 'config'
USER_DATA_FILE = os.path.join(CONFIG_FOLDER, 'user_data.yaml')
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CONFIG_FOLDER, exist_ok=True)

# Global variable to store current dataset
current_df = None

# ============================================
# HELPER FUNCTIONS
# ============================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_users():
    """Load users from YAML file"""
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('users', []) if data else []
    return []

def save_users(users):
    """Save users to YAML file"""
    with open(USER_DATA_FILE, 'w') as f:
        yaml.dump({'users': users}, f, default_flow_style=False)

def get_user_by_email(email):
    """Get user by email"""
    users = load_users()
    return next((u for u in users if u['email'] == email), None)

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

def clean_dataframe(df):
    """Clean and validate the employee dataframe"""
    # Convert date columns
    date_columns = ['joining_date', 'start_date', 'end_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Convert numeric columns
    if 'percentage_allocation' in df.columns:
        df['percentage_allocation'] = pd.to_numeric(df['percentage_allocation'], errors='coerce')

    if 'FTE' in df.columns:
        df['FTE'] = pd.to_numeric(df['FTE'], errors='coerce')

    # Fill missing values
    df = df.fillna({'percentage_allocation': 0, 'FTE': 0, 'project_name': '', 'status': ''})

    return df

def apply_date_filter(df, start_date=None, end_date=None):
    """Apply date filter to dataframe"""
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df = df[
            (df['start_date'] <= end_date) &
            ((df['end_date'] >= start_date) | (df['end_date'].isna()))
        ]
    return df

# ============================================
# AUTHENTICATION ROUTES
# ============================================

@app.route('/')
def index():
    if 'user_email' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """Handle user signup"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        # Validation
        if not username or not email or not password:
            return jsonify({'error': 'All fields are required'}), 400

        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400

        # Check if user exists
        if get_user_by_email(email):
            return jsonify({'error': 'Email already registered'}), 400

        # Create new user
        users = load_users()
        new_user = {
            'username': username,
            'email': email,
            'password': generate_password_hash(password),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        users.append(new_user)
        save_users(users)

        return jsonify({'message': 'Account created successfully'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Handle user login"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        # Find user
        user = get_user_by_email(email)
        if not user or not check_password_hash(user['password'], password):
            return jsonify({'error': 'Invalid email or password'}), 401

        # Set session
        session['user_email'] = user['email']
        session['username'] = user['username']

        return jsonify({'message': 'Login successful', 'username': user['username']}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Handle user logout"""
    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/api/auth/session')
def check_session():
    """Check current session"""
    if 'user_email' in session:
        return jsonify({
            'logged_in': True,
            'username': session.get('username', ''),
            'email': session['user_email']
        }), 200
    return jsonify({'logged_in': False}), 200

# ============================================
# DASHBOARD ROUTES
# ============================================

@app.route('/dashboard')
@login_required
def dashboard():
    """Render the main dashboard page"""
    return render_template('main.html')

# ============================================
# FILE UPLOAD & DATASET ROUTES
# ============================================

@app.route('/api/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Only Excel files (.xlsx, .xls) are allowed'}), 400

        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Validate file
        try:
            df = pd.read_excel(filepath)
            required_columns = ['employee_id', 'employee_name', 'joining_date', 'grade',
                              'designation', 'competency', 'key_skill', 'base_location']

            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                os.remove(filepath)
                return jsonify({'error': f'Missing required columns: {", ".join(missing_columns)}'}), 400

        except Exception as e:
            os.remove(filepath)
            return jsonify({'error': f'Invalid Excel file: {str(e)}'}), 400

        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'rows': len(df)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset/list')
@login_required
def list_datasets():
    """List all available datasets"""
    try:
        datasets = []

        # Check default dataset
        default_file = os.path.join(DATA_FOLDER, 'employee.xlsx')
        if os.path.exists(default_file):
            stat = os.stat(default_file)
            datasets.append({
                'filename': 'employee.xlsx',
                'size': f'{stat.st_size / 1024:.2f} KB',
                'date': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                'path': default_file
            })

        # Check uploaded files
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                if allowed_file(filename):
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    stat = os.stat(filepath)
                    datasets.append({
                        'filename': filename,
                        'size': f'{stat.st_size / 1024:.2f} KB',
                        'date': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                        'path': filepath
                    })

        return jsonify({'datasets': datasets}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset/load/<filename>')
@login_required
def load_dataset(filename):
    """Load a specific dataset"""
    global current_df

    try:
        # Find file
        filepath = None
        if filename == 'employee.xlsx':
            filepath = os.path.join(DATA_FOLDER, filename)
        else:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(filepath):
            return jsonify({'error': 'Dataset not found'}), 404

        # Load and clean data
        df = pd.read_excel(filepath)
        current_df = clean_dataframe(df)

        return jsonify({
            'message': 'Dataset loaded successfully',
            'rows': len(current_df),
            'columns': list(current_df.columns)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# ANALYTICS API ENDPOINTS
# ============================================

@app.route('/api/bench/kpis')
@login_required
def bench_kpis():
    """Get all KPIs"""
    try:
        if current_df is None or current_df.empty:
            return jsonify({'error': 'No dataset loaded'}), 400

        # Get date parameters
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)

        df = current_df.copy()
        df = apply_date_filter(df, start_date, end_date)

        # 1. Number of unique employees
        num_employees = df['employee_id'].nunique()

        # 2. Under-utilized employees (avg FTE < 0.5) - count unique employees only
        under_utilized = df.groupby('employee_id')['FTE'].mean()
        under_skilled = len(under_utilized[under_utilized < 0.5])

        # 3. Billable vs Non-Billable - count unique employees
        unique_billable = df[df['percentage_allocation'] > 0]['employee_id'].nunique()
        unique_non_billable = df[df['percentage_allocation'] == 0]['employee_id'].nunique()

        if num_employees > 0:
            billable_ratio = round((unique_billable / num_employees) * 100, 1)
        else:
            billable_ratio = 0

        # 4. Overall Utilization %
        overall_utilization = round(df['FTE'].mean() * 100, 1)

        return jsonify({
            'num_employees': num_employees,
            'under_skilled': under_skilled,
            'billable_ratio': billable_ratio,
            'utilization': overall_utilization
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bench/location')
@login_required
def bench_location():
    """Get location-wise employee distribution"""
    try:
        if current_df is None or current_df.empty:
            return jsonify({'error': 'No dataset loaded'}), 400

        # Get date parameters
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)

        df = current_df.copy()
        df = apply_date_filter(df, start_date, end_date)

        location_counts = df.groupby('base_location')['employee_id'].nunique()

        return jsonify({
            'locations': location_counts.index.tolist(),
            'counts': location_counts.values.tolist()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bench/grade')
@login_required
def bench_grade():
    """Get grade-wise employee distribution"""
    try:
        if current_df is None or current_df.empty:
            return jsonify({'error': 'No dataset loaded'}), 400

        # Get date parameters
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)

        df = current_df.copy()
        df = apply_date_filter(df, start_date, end_date)

        grade_counts = df.groupby('grade')['employee_id'].nunique()

        return jsonify({
            'grades': grade_counts.index.tolist(),
            'counts': grade_counts.values.tolist()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bench/skills')
@login_required
def bench_skills():
    """Get skill count distribution (treemap data)"""
    try:
        if current_df is None or current_df.empty:
            return jsonify({'error': 'No dataset loaded'}), 400

        # Get date parameters
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)

        df = current_df.copy()
        df = apply_date_filter(df, start_date, end_date)

        skill_counts = df.groupby('key_skill')['employee_id'].nunique().sort_values(ascending=False).head(15)

        return jsonify({
            'skills': skill_counts.index.tolist(),
            'counts': skill_counts.values.tolist()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bench/bench_skills')
@login_required
def bench_skills_available():
    """Get available skills on bench (date-based filter)"""
    try:
        if current_df is None or current_df.empty:
            return jsonify({'error': 'No dataset loaded'}), 400

        # Get date parameters
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)

        # Filter bench employees (percentage_allocation = 0)
        bench_df = current_df[current_df['percentage_allocation'] == 0].copy()
        bench_df = apply_date_filter(bench_df, start_date, end_date)

        skill_counts = bench_df.groupby('key_skill')['employee_id'].nunique().sort_values(ascending=False).head(10)

        return jsonify({
            'skills': skill_counts.index.tolist(),
            'counts': skill_counts.values.tolist()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bench/status')
@login_required
def bench_status():
    """Get status distribution"""
    try:
        if current_df is None or current_df.empty:
            return jsonify({'error': 'No dataset loaded'}), 400

        # Get date parameters
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)

        df = current_df.copy()
        df = apply_date_filter(df, start_date, end_date)

        # Derive status from percentage_allocation
        df['derived_status'] = df['percentage_allocation'].apply(
            lambda x: 'Billable' if x > 0 else 'Bench'
        )

        status_counts = df['derived_status'].value_counts()

        return jsonify({
            'statuses': status_counts.index.tolist(),
            'counts': status_counts.values.tolist()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bench/reasons')
@login_required
def bench_reasons():
    """Get bench reasons distribution from status column"""
    try:
        if current_df is None or current_df.empty:
            return jsonify({'error': 'No dataset loaded'}), 400

        # Get date parameters
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)

        df = current_df.copy()
        df = apply_date_filter(df, start_date, end_date)

        # Get the latest record per employee
        df_sorted = df.sort_values('start_date')
        latest_records_per_employee = df_sorted.groupby('employee_id').tail(1)

        # Filter only those employees whose latest status contains 'bench'
        bench_latest = latest_records_per_employee[
            latest_records_per_employee['status'].str.contains('bench', case=False, na=False)
        ]

        if bench_latest.empty:
            return jsonify({'reasons': [], 'counts': []}), 200

        # Count unique employees per bench reason (status)
        reason_counts = bench_latest.groupby('status')['employee_id'].nunique().sort_values(ascending=False).head(10)

        return jsonify({
            'reasons': reason_counts.index.tolist(),
            'counts': reason_counts.values.tolist()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bench/timeline')
@login_required
def bench_timeline():
    """Get bench trends by week/month/year"""
    try:
        if current_df is None or current_df.empty:
            return jsonify({'error': 'No dataset loaded'}), 400

        period = request.args.get('period', 'M')  # W for week, M for month, Y for year

        # Get date parameters
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)

        # Filter bench records
        bench_df = current_df[current_df['percentage_allocation'] == 0].copy()
        bench_df = apply_date_filter(bench_df, start_date, end_date)

        df_with_dates = bench_df[bench_df['start_date'].notna()].copy()

        if df_with_dates.empty:
            return jsonify({'dates': [], 'counts': []}), 200

        timeline = df_with_dates.groupby(df_with_dates['start_date'].dt.to_period(period)).size()

        return jsonify({
            'dates': [str(d) for d in timeline.index],
            'counts': timeline.values.tolist()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bench/availability')
@login_required
def bench_availability():
    """Get employees approaching end date (table)"""
    try:
        if current_df is None or current_df.empty:
            return jsonify({'error': 'No dataset loaded'}), 400

        # Get date parameters
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)

        df = current_df.copy()
        df = apply_date_filter(df, start_date, end_date)

        today = pd.Timestamp.now()
        df_with_end = df[df['end_date'].notna()].copy()

        if df_with_end.empty:
            return jsonify({'employees': []}), 200

        # Calculate days remaining
        df_with_end['days_remaining'] = (df_with_end['end_date'] - today).dt.days

        # Filter: ending within next 90 days
        upcoming = df_with_end[
            (df_with_end['days_remaining'] > 0) &
            (df_with_end['days_remaining'] <= 90)
        ].sort_values('days_remaining').head(15)

        employees_data = []
        for _, row in upcoming.iterrows():
            employees_data.append({
                'employee_name': row['employee_name'],
                'grade': row['grade'],
                'key_skill': row['key_skill'],
                'end_date': row['end_date'].strftime('%Y-%m-%d'),
                'days_remaining': int(row['days_remaining'])
            })

        return jsonify({'employees': employees_data}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bench/recommendations')
@login_required
def bench_recommendations():
    """Generate action items and recommendations"""
    try:
        if current_df is None or current_df.empty:
            return jsonify({'error': 'No dataset loaded'}), 400

        # Get date parameters - FIXED: Now uses date filter
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)

        df = current_df.copy()
        df = apply_date_filter(df, start_date, end_date)

        recommendations = []

        # Get the latest record per employee
        df_sorted = df.sort_values('start_date')
        latest_status_per_employee = df_sorted.groupby('employee_id').tail(1)

        # Rule 1: High bench count
        bench_available_df = latest_status_per_employee[
            latest_status_per_employee['status'].str.contains('bench available for opportunity', case=False, na=False)
        ]

        bench_count = bench_available_df['employee_id'].nunique()

        if bench_count > 20:
            recommendations.append({
                'priority': 'HIGH',
                'message': f'{bench_count} employees on bench. Consider accelerating resource allocation.'
            })
        elif bench_count > 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'message': f'{bench_count} employees on bench. Monitor resource allocation.'
            })

        # Rule 2: Under-utilized employees
        under_utilized = df.groupby('employee_id')['FTE'].mean()
        under_skilled_count = len(under_utilized[under_utilized < 0.5])

        if under_skilled_count > 5:
            recommendations.append({
                'priority': 'MEDIUM',
                'message': f'{under_skilled_count} under-utilized employees (avg FTE < 50%). Review allocation strategy.'
            })

        # Rule 3: Upcoming availability
        today = pd.Timestamp.now()
        df_with_end = df[df['end_date'].notna()].copy()

        if not df_with_end.empty:
            df_with_end['days_remaining'] = (df_with_end['end_date'] - today).dt.days
            upcoming_30 = df_with_end[
                (df_with_end['days_remaining'] > 0) &
                (df_with_end['days_remaining'] <= 30)
            ]['employee_id'].nunique()

            if upcoming_30 > 0:
                recommendations.append({
                    'priority': 'HIGH',
                    'message': f'{upcoming_30} employees becoming available in next 30 days. Prepare allocation plan.'
                })

        # Rule 4: Skills concentration
        if not df['key_skill'].empty:
            skill_dist = df.groupby('key_skill')['employee_id'].nunique()
            if len(skill_dist) > 0:
                top_skill = skill_dist.idxmax()
                top_skill_count = skill_dist.max()
                total_unique_employees = df['employee_id'].nunique()

                if total_unique_employees > 0 and top_skill_count > total_unique_employees * 0.3:
                    recommendations.append({
                        'priority': 'LOW',
                        'message': f'{top_skill} dominates with {top_skill_count} employees. Consider skill diversification.'
                    })

        # If no specific recommendations
        if not recommendations:
            recommendations.append({
                'priority': 'INFO',
                'message': 'All metrics within normal range. Continue monitoring resource allocation.'
            })

        return jsonify({'recommendations': recommendations}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 25 MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================
# RUN APPLICATION
# ============================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)